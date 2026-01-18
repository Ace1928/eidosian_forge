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
def apply_entry(self, entry):
    try:
        entry()
    except Exception as exc:
        if not self.handle_error(exc):
            logger.error('Error in timer: %r', exc, exc_info=True)