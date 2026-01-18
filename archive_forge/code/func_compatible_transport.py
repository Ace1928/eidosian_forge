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
def compatible_transport(self, app):
    with app.connection_for_read() as conn:
        return conn.transport.driver_type in self.compatible_transports