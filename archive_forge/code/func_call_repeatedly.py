from __future__ import annotations
import heapq
import sys
from collections import namedtuple
from datetime import datetime
from functools import total_ordering
from time import monotonic
from time import time as _time
from typing import TYPE_CHECKING
from weakref import proxy as weakrefproxy
from vine.utils import wraps
from kombu.log import get_logger
def call_repeatedly(self, secs, fun, args=(), kwargs=None, priority=0):
    kwargs = {} if not kwargs else kwargs
    tref = self.Entry(fun, args, kwargs)

    @wraps(fun)
    def _reschedules(*args, **kwargs):
        last, now = (tref._last_run, monotonic())
        lsince = now - tref._last_run if last else secs
        try:
            if lsince and lsince >= secs:
                tref._last_run = now
                return fun(*args, **kwargs)
        finally:
            if not tref.canceled:
                last = tref._last_run
                next = secs - (now - last) if last else secs
                self.enter_after(next, tref, priority)
    tref.fun = _reschedules
    tref._last_run = None
    return self.enter_after(secs, tref, priority)