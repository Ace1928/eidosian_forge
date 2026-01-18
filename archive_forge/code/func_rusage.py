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
def rusage(self):
    if resource is None:
        raise NotImplementedError('rusage not supported by this platform')
    s = resource.getrusage(resource.RUSAGE_SELF)
    return {'utime': s.ru_utime, 'stime': s.ru_stime, 'maxrss': s.ru_maxrss, 'ixrss': s.ru_ixrss, 'idrss': s.ru_idrss, 'isrss': s.ru_isrss, 'minflt': s.ru_minflt, 'majflt': s.ru_majflt, 'nswap': s.ru_nswap, 'inblock': s.ru_inblock, 'oublock': s.ru_oublock, 'msgsnd': s.ru_msgsnd, 'msgrcv': s.ru_msgrcv, 'nsignals': s.ru_nsignals, 'nvcsw': s.ru_nvcsw, 'nivcsw': s.ru_nivcsw}