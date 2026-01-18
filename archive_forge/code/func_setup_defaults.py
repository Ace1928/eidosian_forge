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
def setup_defaults(self, concurrency=None, loglevel='WARN', logfile=None, task_events=None, pool=None, consumer_cls=None, timer_cls=None, timer_precision=None, autoscaler_cls=None, pool_putlocks=None, pool_restarts=None, optimization=None, O=None, statedb=None, time_limit=None, soft_time_limit=None, scheduler=None, pool_cls=None, state_db=None, task_time_limit=None, task_soft_time_limit=None, scheduler_cls=None, schedule_filename=None, max_tasks_per_child=None, prefetch_multiplier=None, disable_rate_limits=None, worker_lost_wait=None, max_memory_per_child=None, **_kw):
    either = self.app.either
    self.loglevel = loglevel
    self.logfile = logfile
    self.concurrency = either('worker_concurrency', concurrency)
    self.task_events = either('worker_send_task_events', task_events)
    self.pool_cls = either('worker_pool', pool, pool_cls)
    self.consumer_cls = either('worker_consumer', consumer_cls)
    self.timer_cls = either('worker_timer', timer_cls)
    self.timer_precision = either('worker_timer_precision', timer_precision)
    self.optimization = optimization or O
    self.autoscaler_cls = either('worker_autoscaler', autoscaler_cls)
    self.pool_putlocks = either('worker_pool_putlocks', pool_putlocks)
    self.pool_restarts = either('worker_pool_restarts', pool_restarts)
    self.statedb = either('worker_state_db', statedb, state_db)
    self.schedule_filename = either('beat_schedule_filename', schedule_filename)
    self.scheduler = either('beat_scheduler', scheduler, scheduler_cls)
    self.time_limit = either('task_time_limit', time_limit, task_time_limit)
    self.soft_time_limit = either('task_soft_time_limit', soft_time_limit, task_soft_time_limit)
    self.max_tasks_per_child = either('worker_max_tasks_per_child', max_tasks_per_child)
    self.max_memory_per_child = either('worker_max_memory_per_child', max_memory_per_child)
    self.prefetch_multiplier = int(either('worker_prefetch_multiplier', prefetch_multiplier))
    self.disable_rate_limits = either('worker_disable_rate_limits', disable_rate_limits)
    self.worker_lost_wait = either('worker_lost_wait', worker_lost_wait)