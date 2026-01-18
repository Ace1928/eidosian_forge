import logging
import sys
from datetime import datetime
from time import monotonic, time
from weakref import ref
from billiard.common import TERM_SIGNAME
from billiard.einfo import ExceptionWithTraceback
from kombu.utils.encoding import safe_repr, safe_str
from kombu.utils.objects import cached_property
from celery import current_app, signals
from celery.app.task import Context
from celery.app.trace import fast_trace_task, trace_task, trace_task_ret
from celery.concurrency.base import BasePool
from celery.exceptions import (Ignore, InvalidTaskError, Reject, Retry, TaskRevokedError, Terminated,
from celery.platforms import signals as _signals
from celery.utils.functional import maybe, maybe_list, noop
from celery.utils.log import get_logger
from celery.utils.nodenames import gethostname
from celery.utils.serialization import get_pickled_exception
from celery.utils.time import maybe_iso8601, maybe_make_aware, timezone
from . import state
def maybe_expire(self):
    """If expired, mark the task as revoked."""
    if self.expires:
        now = datetime.now(self.expires.tzinfo)
        if now > self.expires:
            revoked_tasks.add(self.id)
            return True