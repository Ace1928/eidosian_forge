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
def handle_error_state(self, task, req, eager=False, call_errbacks=True):
    if task.ignore_result:
        store_errors = task.store_errors_even_if_ignored
    elif eager and task.store_eager_result:
        store_errors = True
    else:
        store_errors = not eager
    return {RETRY: self.handle_retry, FAILURE: self.handle_failure}[self.state](task, req, store_errors=store_errors, call_errbacks=call_errbacks)