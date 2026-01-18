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
def set_in_sighandler(value):
    """Set flag signifiying that we're inside a signal handler."""
    global _in_sighandler
    _in_sighandler = value