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
def ensure_started(self):
    """Wait for worker to be fully up and running.

        Warning:
            Worker must be started within a thread for this to work,
            or it will block forever.
        """
    self._on_started.wait()