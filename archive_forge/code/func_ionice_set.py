import contextlib
import errno
import functools
import os
import signal
import sys
import time
from collections import namedtuple
from . import _common
from ._common import ENCODING
from ._common import ENCODING_ERRS
from ._common import AccessDenied
from ._common import NoSuchProcess
from ._common import TimeoutExpired
from ._common import conn_tmap
from ._common import conn_to_ntuple
from ._common import debug
from ._common import isfile_strict
from ._common import memoize
from ._common import memoize_when_activated
from ._common import parse_environ_block
from ._common import usage_percent
from ._compat import PY3
from ._compat import long
from ._compat import lru_cache
from ._compat import range
from ._compat import unicode
from ._psutil_windows import ABOVE_NORMAL_PRIORITY_CLASS
from ._psutil_windows import BELOW_NORMAL_PRIORITY_CLASS
from ._psutil_windows import HIGH_PRIORITY_CLASS
from ._psutil_windows import IDLE_PRIORITY_CLASS
from ._psutil_windows import NORMAL_PRIORITY_CLASS
from ._psutil_windows import REALTIME_PRIORITY_CLASS
@wrap_exceptions
def ionice_set(self, ioclass, value):
    if value:
        msg = 'value argument not accepted on Windows'
        raise TypeError(msg)
    if ioclass not in (IOPRIO_VERYLOW, IOPRIO_LOW, IOPRIO_NORMAL, IOPRIO_HIGH):
        raise ValueError('%s is not a valid priority' % ioclass)
    cext.proc_io_priority_set(self.pid, ioclass)