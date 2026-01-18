from __future__ import division
import sys as _sys
import datetime as _datetime
import uuid as _uuid
import traceback as _traceback
import os as _os
import logging as _logging
from syslog import (LOG_EMERG, LOG_ALERT, LOG_CRIT, LOG_ERR,
from ._journal import __version__, sendv, stream_fd
from ._reader import (_Reader, NOP, APPEND, INVALIDATE,
from . import id128 as _id128
@staticmethod
def map_priority(levelno):
    """Map logging levels to journald priorities.

        Since Python log level numbers are "sparse", we have to map numbers in
        between the standard levels too.
        """
    if levelno <= _logging.DEBUG:
        return LOG_DEBUG
    elif levelno <= _logging.INFO:
        return LOG_INFO
    elif levelno <= _logging.WARNING:
        return LOG_WARNING
    elif levelno <= _logging.ERROR:
        return LOG_ERR
    elif levelno <= _logging.CRITICAL:
        return LOG_CRIT
    else:
        return LOG_ALERT