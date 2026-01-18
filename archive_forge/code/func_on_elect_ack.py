from collections import defaultdict
from functools import partial
from heapq import heappush
from operator import itemgetter
from kombu import Consumer
from kombu.asynchronous.semaphore import DummyLock
from kombu.exceptions import ContentDisallowed, DecodeError
from celery import bootsteps
from celery.utils.log import get_logger
from celery.utils.objects import Bunch
from .mingle import Mingle
def on_elect_ack(self, event):
    id = event['id']
    try:
        replies = self.consensus_replies[id]
    except KeyError:
        return
    alive_workers = set(self.state.alive_workers())
    replies.append(event['hostname'])
    if len(replies) >= len(alive_workers):
        _, leader, topic, action = self.clock.sort_heap(self.consensus_requests[id])
        if leader == self.full_hostname:
            info('I won the election %r', id)
            try:
                handler = self.election_handlers[topic]
            except KeyError:
                logger.exception('Unknown election topic %r', topic)
            else:
                handler(action)
        else:
            info('node %s elected for %r', leader, id)
        self.consensus_requests.pop(id, None)
        self.consensus_replies.pop(id, None)