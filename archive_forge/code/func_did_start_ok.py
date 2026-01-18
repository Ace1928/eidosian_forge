import os
from billiard import forking_enable
from billiard.common import REMAP_SIGTERM, TERM_SIGNAME
from billiard.pool import CLOSE, RUN
from billiard.pool import Pool as BlockingPool
from celery import platforms, signals
from celery._state import _set_task_join_will_block, set_default_app
from celery.app import trace
from celery.concurrency.base import BasePool
from celery.utils.functional import noop
from celery.utils.log import get_logger
from .asynpool import AsynPool
def did_start_ok(self):
    return self._pool.did_start_ok()