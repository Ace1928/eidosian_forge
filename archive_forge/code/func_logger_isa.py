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
def logger_isa(l, p, max=1000):
    this, seen = (l, set())
    for _ in range(max):
        if this == p:
            return True
        else:
            if this in seen:
                raise RuntimeError(f'Logger {l.name!r} parents recursive')
            seen.add(this)
            this = this.parent
            if not this:
                break
    else:
        raise RuntimeError(f'Logger hierarchy exceeds {max}')
    return False