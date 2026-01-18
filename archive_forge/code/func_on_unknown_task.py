from the broker, processing the messages and keeping the broker connections
import errno
import logging
import os
import warnings
from collections import defaultdict
from time import sleep
from billiard.common import restart_state
from billiard.exceptions import RestartFreqExceeded
from kombu.asynchronous.semaphore import DummyLock
from kombu.exceptions import ContentDisallowed, DecodeError
from kombu.utils.compat import _detect_environment
from kombu.utils.encoding import safe_repr
from kombu.utils.limits import TokenBucket
from vine import ppartial, promise
from celery import bootsteps, signals
from celery.app.trace import build_tracer
from celery.exceptions import (CPendingDeprecationWarning, InvalidTaskError, NotRegistered, WorkerShutdown,
from celery.utils.functional import noop
from celery.utils.log import get_logger
from celery.utils.nodenames import gethostname
from celery.utils.objects import Bunch
from celery.utils.text import truncate
from celery.utils.time import humanize_seconds, rate
from celery.worker import loops
from celery.worker.state import active_requests, maybe_shutdown, requests, reserved_requests, task_reserved
def on_unknown_task(self, body, message, exc):
    error(UNKNOWN_TASK_ERROR, exc, dump_body(message, body), message.headers, message.delivery_info, exc_info=True)
    try:
        id_, name = (message.headers['id'], message.headers['task'])
        root_id = message.headers.get('root_id')
    except KeyError:
        payload = message.payload
        id_, name = (payload['id'], payload['task'])
        root_id = None
    request = Bunch(name=name, chord=None, root_id=root_id, correlation_id=message.properties.get('correlation_id'), reply_to=message.properties.get('reply_to'), errbacks=None)
    message.reject_log_error(logger, self.connection_errors)
    self.app.backend.mark_as_failure(id_, NotRegistered(name), request=request)
    if self.event_dispatcher:
        self.event_dispatcher.send('task-failed', uuid=id_, exception=f'NotRegistered({name!r})')
    signals.task_unknown.send(sender=self, message=message, exc=exc, name=name, id=id_)