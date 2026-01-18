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
def query_unique(self, field):
    """Return a list of unique values appearing in the journal for the given
        `field`.

        Note this does not respect any journal matches.

        Entries will be processed with converters specified during
        Reader creation.
        """
    return set((self._convert_field(field, value) for value in super(Reader, self).query_unique(field)))