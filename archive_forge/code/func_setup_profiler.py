import atexit
import os
import platform
import random
import sys
import threading
import time
import uuid
from collections import deque
import sentry_sdk
from sentry_sdk._compat import PY33, PY311
from sentry_sdk._lru_cache import LRUCache
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
def setup_profiler(options):
    global _scheduler
    if _scheduler is not None:
        logger.debug('[Profiling] Profiler is already setup')
        return False
    if not PY33:
        logger.warn('[Profiling] Profiler requires Python >= 3.3')
        return False
    frequency = DEFAULT_SAMPLING_FREQUENCY
    if is_gevent():
        default_profiler_mode = GeventScheduler.mode
    else:
        default_profiler_mode = ThreadScheduler.mode
    if options.get('profiler_mode') is not None:
        profiler_mode = options['profiler_mode']
    else:
        profiler_mode = options.get('_experiments', {}).get('profiler_mode') or default_profiler_mode
    if profiler_mode == ThreadScheduler.mode or profiler_mode == 'sleep':
        _scheduler = ThreadScheduler(frequency=frequency)
    elif profiler_mode == GeventScheduler.mode:
        _scheduler = GeventScheduler(frequency=frequency)
    else:
        raise ValueError('Unknown profiler mode: {}'.format(profiler_mode))
    logger.debug('[Profiling] Setting up profiler in {mode} mode'.format(mode=_scheduler.mode))
    _scheduler.setup()
    atexit.register(teardown_profiler)
    return True