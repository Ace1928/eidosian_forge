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
def enter_at(self, entry, eta=None, priority=0, time=monotonic):
    """Enter function into the scheduler.

        Arguments:
        ---------
            entry (~kombu.asynchronous.timer.Entry): Item to enter.
            eta (datetime.datetime): Scheduled time.
            priority (int): Unused.
        """
    if eta is None:
        eta = time()
    if isinstance(eta, datetime):
        try:
            eta = to_timestamp(eta)
        except Exception as exc:
            if not self.handle_error(exc):
                raise
            return
    return self._enter(eta, priority, entry)