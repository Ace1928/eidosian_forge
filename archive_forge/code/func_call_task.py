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
def call_task(self, task):
    try:
        self.app.signature(task).apply_async()
    except Exception as exc:
        logger.exception('Could not call task: %r', exc)