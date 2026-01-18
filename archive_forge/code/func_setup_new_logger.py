import os
import sys
import signal
import threading
import logging
from dataclasses import dataclass
from functools import partialmethod
from typing import Optional, Any, Dict, List
from .mp_utils import _CPU_CORES, _MAX_THREADS, _MAX_PROCS
def setup_new_logger(name, log_level='info', quiet_loggers=None, clear_handlers=False, propagate=True):
    logger_config = {'name': name, 'log_level': log_level, 'console_log_output': 'stdout', 'console_log_level': 'info', 'console_log_color': True, 'logfile_file': None, 'logfile_log_level': 'debug', 'logfile_log_color': False, 'log_line_template': f'%(color_on)s[{name}] %(name)-5s %(funcName)-5s%(color_off)s: %(message)s', 'clear_handlers': clear_handlers, 'quiet_loggers': quiet_loggers, 'propagate': propagate}
    return LazyOpsLogger(logger_config)