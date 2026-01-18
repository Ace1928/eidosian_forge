import time
from operator import itemgetter
from kombu import Queue
from kombu.connection import maybe_channel
from kombu.mixins import ConsumerMixin
from celery import uuid
from celery.app import app_or_default
from celery.utils.time import adjust_timestamp
from .event import get_exchange
def event_from_message(self, body, localize=True, now=time.time, tzfields=_TZGETTER, adjust_timestamp=adjust_timestamp, CLIENT_CLOCK_SKEW=CLIENT_CLOCK_SKEW):
    type = body['type']
    if type == 'task-sent':
        _c = body['clock'] = (self.clock.value or 1) + CLIENT_CLOCK_SKEW
        self.adjust_clock(_c)
    else:
        try:
            clock = body['clock']
        except KeyError:
            body['clock'] = self.forward_clock()
        else:
            self.adjust_clock(clock)
    if localize:
        try:
            offset, timestamp = tzfields(body)
        except KeyError:
            pass
        else:
            body['timestamp'] = adjust_timestamp(timestamp, offset)
    body['local_received'] = now()
    return (type, body)