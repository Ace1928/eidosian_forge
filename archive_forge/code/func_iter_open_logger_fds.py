import logging
import numbers
import os
import sys
import threading
import traceback
from contextlib import contextmanager
from typing import AnyStr, Sequence  # noqa
from kombu.log import LOG_LEVELS
from kombu.log import get_logger as _get_logger
from kombu.utils.encoding import safe_str
from .term import colored
def iter_open_logger_fds():
    seen = set()
    loggers = list(logging.Logger.manager.loggerDict.values()) + [logging.getLogger(None)]
    for l in loggers:
        try:
            for handler in l.handlers:
                try:
                    if handler not in seen:
                        yield handler.stream
                        seen.add(handler)
                except AttributeError:
                    pass
        except AttributeError:
            pass