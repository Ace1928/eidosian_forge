import os
import sys
import signal
import threading
import logging
from dataclasses import dataclass
from functools import partialmethod
from typing import Optional, Any, Dict, List
from .mp_utils import _CPU_CORES, _MAX_THREADS, _MAX_PROCS
@classmethod
def enable_watcher(cls):
    if EnvChecker.watcher_enabled:
        return
    EnvChecker.sigs['sigint'] = signal.signal(signal.SIGINT, EnvChecker.exit_handler)
    EnvChecker.sigs['sigterm'] = signal.signal(signal.SIGTERM, EnvChecker.exit_handler)
    EnvChecker.loggers['LazyWatch'] = EnvChecker.get_logger(name='LazyWatch')
    EnvChecker.watcher_enabled = True