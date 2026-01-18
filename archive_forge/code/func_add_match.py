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
def add_match(self, *args, **kwargs):
    """Add one or more matches to the filter journal log entries.

        All matches of different field are combined with logical AND, and
        matches of the same field are automatically combined with logical OR.
        Matches can be passed as strings of form "FIELD=value", or keyword
        arguments FIELD="value".
        """
    args = list(args)
    args.extend((_make_line(key, val) for key, val in kwargs.items()))
    for arg in args:
        super(Reader, self).add_match(arg)