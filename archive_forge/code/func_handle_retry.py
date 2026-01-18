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
def handle_retry(self, task, req, store_errors=True, **kwargs):
    """Handle retry exception."""
    type_, _, tb = sys.exc_info()
    try:
        reason = self.retval
        einfo = ExceptionInfo((type_, reason, tb))
        if store_errors:
            task.backend.mark_as_retry(req.id, reason.exc, einfo.traceback, request=req)
        task.on_retry(reason.exc, req.id, req.args, req.kwargs, einfo)
        signals.task_retry.send(sender=task, request=req, reason=reason, einfo=einfo)
        info(LOG_RETRY, {'id': req.id, 'name': get_task_name(req, task.name), 'exc': str(reason)})
        return einfo
    finally:
        del tb