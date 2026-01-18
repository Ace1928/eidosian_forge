import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterable, Optional, Union
import celery.worker.consumer  # noqa
from celery import Celery, worker
from celery.result import _set_task_join_will_block, allow_join_result
from celery.utils.dispatch import Signal
from celery.utils.nodenames import anon_nodename
def on_consumer_ready(self, consumer):
    """Callback called when the Consumer blueprint is fully started."""
    self._on_started.set()
    test_worker_started.send(sender=self.app, worker=self, consumer=consumer)