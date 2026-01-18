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
def seek_realtime(self, realtime):
    """Seek to a matching journal entry nearest to `timestamp` time.

        Argument `realtime` must be either an integer UNIX timestamp (in
        microseconds since the beginning of the UNIX epoch), or an float UNIX
        timestamp (in seconds since the beginning of the UNIX epoch), or a
        datetime.datetime instance. The integer form is deprecated.

        >>> import time
        >>> from systemd import journal

        >>> yesterday = time.time() - 24 * 60**2
        >>> j = journal.Reader()
        >>> j.seek_realtime(yesterday)
        """
    if isinstance(realtime, _datetime.datetime):
        try:
            realtime = realtime.astimezone()
        except TypeError:
            pass
        realtime = int(float(realtime.strftime('%s.%f')) * 1000000)
    elif not isinstance(realtime, int):
        realtime = int(realtime * 1000000)
    return super(Reader, self).seek_realtime(realtime)