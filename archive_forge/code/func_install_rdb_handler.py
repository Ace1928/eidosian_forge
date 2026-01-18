import logging
import os
import platform as _platform
import sys
from datetime import datetime
from functools import partial
from billiard.common import REMAP_SIGTERM
from billiard.process import current_process
from kombu.utils.encoding import safe_str
from celery import VERSION_BANNER, platforms, signals
from celery.app import trace
from celery.loaders.app import AppLoader
from celery.platforms import EX_FAILURE, EX_OK, check_privileges
from celery.utils import static, term
from celery.utils.debug import cry
from celery.utils.imports import qualname
from celery.utils.log import get_logger, in_sighandler, set_in_sighandler
from celery.utils.text import pluralize
from celery.worker import WorkController
def install_rdb_handler(envvar='CELERY_RDBSIG', sig='SIGUSR2'):

    def rdb_handler(*args):
        """Signal handler setting a rdb breakpoint at the current frame."""
        with in_sighandler():
            from celery.contrib.rdb import _frame, set_trace
            frame = args[1] if args else _frame().f_back
            set_trace(frame)
    if os.environ.get(envvar):
        platforms.signals[sig] = rdb_handler