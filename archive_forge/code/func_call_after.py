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
def call_after(self, secs, fun, args=(), kwargs=None, priority=0):
    kwargs = {} if not kwargs else kwargs
    return self.enter_after(secs, self.Entry(fun, args, kwargs), priority)