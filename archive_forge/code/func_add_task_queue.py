from the broker, processing the messages and keeping the broker connections
import errno
import logging
import os
import warnings
from collections import defaultdict
from time import sleep
from billiard.common import restart_state
from billiard.exceptions import RestartFreqExceeded
from kombu.asynchronous.semaphore import DummyLock
from kombu.exceptions import ContentDisallowed, DecodeError
from kombu.utils.compat import _detect_environment
from kombu.utils.encoding import safe_repr
from kombu.utils.limits import TokenBucket
from vine import ppartial, promise
from celery import bootsteps, signals
from celery.app.trace import build_tracer
from celery.exceptions import (CPendingDeprecationWarning, InvalidTaskError, NotRegistered, WorkerShutdown,
from celery.utils.functional import noop
from celery.utils.log import get_logger
from celery.utils.nodenames import gethostname
from celery.utils.objects import Bunch
from celery.utils.text import truncate
from celery.utils.time import humanize_seconds, rate
from celery.worker import loops
from celery.worker.state import active_requests, maybe_shutdown, requests, reserved_requests, task_reserved
def add_task_queue(self, queue, exchange=None, exchange_type=None, routing_key=None, **options):
    cset = self.task_consumer
    queues = self.app.amqp.queues
    if queue in queues:
        q = queues[queue]
    else:
        exchange = queue if exchange is None else exchange
        exchange_type = 'direct' if exchange_type is None else exchange_type
        q = queues.select_add(queue, exchange=exchange, exchange_type=exchange_type, routing_key=routing_key, **options)
    if not cset.consuming_from(queue):
        cset.add_queue(q)
        cset.consume()
        info('Started consuming from %s', queue)