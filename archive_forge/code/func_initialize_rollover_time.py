import datetime
import errno
import logging
import os
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from io import TextIOWrapper
from logging.handlers import BaseRotatingHandler, TimedRotatingFileHandler
from typing import TYPE_CHECKING, Dict, Generator, List, Optional, Tuple
from portalocker import LOCK_EX, lock, unlock
import logging.handlers  # noqa: E402
def initialize_rollover_time(self) -> None:
    """Run by the __init__ to read an existing rollover time from the lockfile,
        and if it can't do that, compute and write a new one."""
    try:
        self.clh._do_lock()
        self.read_rollover_time()
        self._console_log(f'Initializing; reading rollover time: {self.rolloverAt}')
        if self.rolloverAt != 0:
            return
        current_time = int(time.time())
        new_rollover_at = self.computeRollover(current_time)
        while new_rollover_at <= current_time:
            new_rollover_at += self.interval
        self.rolloverAt = new_rollover_at
        self.write_rollover_time()
        self._console_log(f'Set initial rollover time: {self.rolloverAt}')
    finally:
        self.clh._do_unlock()