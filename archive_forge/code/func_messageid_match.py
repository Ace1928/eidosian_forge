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
def messageid_match(self, messageid):
    """Add match for log entries with specified `messageid`.

        `messageid` can be string of hexadicimal digits or a UUID
        instance. Standard message IDs can be found in systemd.id128.

        Equivalent to add_match(MESSAGE_ID=`messageid`).
        """
    if isinstance(messageid, _uuid.UUID):
        messageid = messageid.hex
    self.add_match(MESSAGE_ID=messageid)