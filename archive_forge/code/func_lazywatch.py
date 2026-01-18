import os
import sys
import signal
import threading
import logging
from dataclasses import dataclass
from functools import partialmethod
from typing import Optional, Any, Dict, List
from .mp_utils import _CPU_CORES, _MAX_THREADS, _MAX_PROCS
def lazywatch(name, *args, **kwargs):

    def decorator(func):

        def wrapper(*args, **kwargs):
            logger.info(f'Adding {name} to Exit Handlers')
            LazyEnv.add_exit_handler(name, func)
            return func(*args, **kwargs)
        return wrapper
    return decorator