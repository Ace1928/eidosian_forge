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
def do_gzip(self, input_filename: str) -> None:
    if not gzip:
        self._console_log('#no gzip available', stack=False)
        return
    out_filename = input_filename + '.gz'
    with open(input_filename, 'rb') as input_fh, gzip.open(out_filename, 'wb') as gzip_fh:
        while True:
            data = input_fh.read(self.gzip_buffer)
            if not data:
                break
            gzip_fh.write(data)
    os.remove(input_filename)
    self._console_log(f'#gzipped: {out_filename}', stack=False)