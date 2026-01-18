import os
import sys
from datetime import datetime, timezone
from billiard import cpu_count
from kombu.utils.compat import detect_environment
from celery import bootsteps
from celery import concurrency as _concurrency
from celery import signals
from celery.bootsteps import RUN, TERMINATE
from celery.exceptions import ImproperlyConfigured, TaskRevokedError, WorkerTerminate
from celery.platforms import EX_FAILURE, create_pidlock
from celery.utils.imports import reload_from_cwd
from celery.utils.log import mlevel
from celery.utils.log import worker_logger as logger
from celery.utils.nodenames import default_nodename, worker_direct
from celery.utils.text import str_to_list
from celery.utils.threads import default_socket_timeout
from . import state
def setup_queues(self, include, exclude=None):
    include = str_to_list(include)
    exclude = str_to_list(exclude)
    try:
        self.app.amqp.queues.select(include)
    except KeyError as exc:
        raise ImproperlyConfigured(SELECT_UNKNOWN_QUEUE.strip().format(include, exc))
    try:
        self.app.amqp.queues.deselect(exclude)
    except KeyError as exc:
        raise ImproperlyConfigured(DESELECT_UNKNOWN_QUEUE.strip().format(exclude, exc))
    if self.app.conf.worker_direct:
        self.app.amqp.queues.select_add(worker_direct(self.hostname))