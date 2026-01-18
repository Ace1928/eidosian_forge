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
def cpu_affinity_set(self, value):

    def to_bitmask(ls):
        if not ls:
            raise ValueError('invalid argument %r' % ls)
        out = 0
        for b in ls:
            out |= 2 ** b
        return out
    allcpus = list(range(len(per_cpu_times())))
    for cpu in value:
        if cpu not in allcpus:
            if not isinstance(cpu, (int, long)):
                raise TypeError('invalid CPU %r; an integer is required' % cpu)
            else:
                raise ValueError('invalid CPU %r' % cpu)
    bitmask = to_bitmask(value)
    cext.proc_cpu_affinity_set(self.pid, bitmask)