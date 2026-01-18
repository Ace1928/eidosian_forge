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
@staticmethod
def baseLockFilename(baseFilename: str) -> Tuple[str, str]:
    lock_file = baseFilename[:-4] if baseFilename.endswith('.log') else baseFilename
    lock_file += '.lock'
    lock_path, lock_name = os.path.split(lock_file)
    return (lock_path, '.__' + lock_name)