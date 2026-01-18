import os
import threading
from time import monotonic, sleep
from kombu.asynchronous.semaphore import DummyLock
from celery import bootsteps
from celery.utils.log import get_logger
from celery.utils.threads import bgThread
from . import state
from .components import Pool
def maybe_scale(self, req=None):
    if self._maybe_scale(req):
        self.pool.maintain_pool()