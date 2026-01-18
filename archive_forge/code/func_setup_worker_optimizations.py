import logging
import os
import sys
import time
from collections import namedtuple
from warnings import warn
from billiard.einfo import ExceptionInfo, ExceptionWithTraceback
from kombu.exceptions import EncodeError
from kombu.serialization import loads as loads_message
from kombu.serialization import prepare_accept_content
from kombu.utils.encoding import safe_repr, safe_str
from celery import current_app, group, signals, states
from celery._state import _task_stack
from celery.app.task import Context
from celery.app.task import Task as BaseTask
from celery.exceptions import BackendGetMetaError, Ignore, InvalidTaskError, Reject, Retry
from celery.result import AsyncResult
from celery.utils.log import get_logger
from celery.utils.nodenames import gethostname
from celery.utils.objects import mro_lookup
from celery.utils.saferepr import saferepr
from celery.utils.serialization import get_pickleable_etype, get_pickleable_exception, get_pickled_exception
from celery.worker.state import successful_requests
def setup_worker_optimizations(app, hostname=None):
    """Setup worker related optimizations."""
    hostname = hostname or gethostname()
    _install_stack_protection()
    app.set_current()
    app.set_default()
    app.finalize()
    _localized[:] = [app._tasks, prepare_accept_content(app.conf.accept_content), hostname]
    app.use_fast_trace_task = True