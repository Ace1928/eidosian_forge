import copy
import errno
import itertools
import os
import platform
import signal
import sys
import threading
import time
import warnings
from collections import deque
from functools import partial
from . import cpu_count, get_context
from . import util
from .common import (
from .compat import get_errno, mem_rss, send_offset
from .einfo import ExceptionInfo
from .dummy import DummyProcess
from .exceptions import (
from time import monotonic
from queue import Queue, Empty
from .util import Finalize, debug, warning
def handle_timeouts(self):
    t_hard, t_soft = (self.t_hard, self.t_soft)
    dirty = set()
    on_soft_timeout = self.on_soft_timeout
    on_hard_timeout = self.on_hard_timeout

    def _timed_out(start, timeout):
        if not start or not timeout:
            return False
        if monotonic() >= start + timeout:
            return True
    while self._state == RUN:
        cache = copy.copy(self.cache)
        if dirty:
            dirty = set((k for k in dirty if k in cache))
        for i, job in cache.items():
            ack_time = job._time_accepted
            soft_timeout = job._soft_timeout
            if soft_timeout is None:
                soft_timeout = t_soft
            hard_timeout = job._timeout
            if hard_timeout is None:
                hard_timeout = t_hard
            if _timed_out(ack_time, hard_timeout):
                on_hard_timeout(job)
            elif i not in dirty and _timed_out(ack_time, soft_timeout):
                on_soft_timeout(job)
                dirty.add(i)
        yield