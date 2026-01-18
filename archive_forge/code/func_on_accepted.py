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
def on_accepted(self, pid, time_accepted):
    """Handler called when task is accepted by worker pool."""
    self.worker_pid = pid
    self.time_start = time() - (monotonic() - time_accepted)
    task_accepted(self)
    if not self.task.acks_late:
        self.acknowledge()
    self.send_event('task-started')
    if _does_debug:
        debug('Task accepted: %s[%s] pid:%r', self.name, self.id, pid)
    if self._terminate_on_ack is not None:
        self.terminate(*self._terminate_on_ack)