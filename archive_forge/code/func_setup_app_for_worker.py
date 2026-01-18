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
def setup_app_for_worker(app: Celery, loglevel: Union[str, int], logfile: str) -> None:
    """Setup the app to be used for starting an embedded worker."""
    app.finalize()
    app.set_current()
    app.set_default()
    type(app.log)._setup = False
    app.log.setup(loglevel=loglevel, logfile=logfile)