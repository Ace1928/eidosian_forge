import copy
import errno
import heapq
import os
import shelve
import sys
import time
import traceback
from calendar import timegm
from collections import namedtuple
from functools import total_ordering
from threading import Event, Thread
from billiard import ensure_multiprocessing
from billiard.common import reset_signals
from billiard.context import Process
from kombu.utils.functional import maybe_evaluate, reprcall
from kombu.utils.objects import cached_property
from . import __version__, platforms, signals
from .exceptions import reraise
from .schedules import crontab, maybe_schedule
from .utils.functional import is_numeric_value
from .utils.imports import load_extension_class_names, symbol_by_name
from .utils.log import get_logger, iter_open_logger_fds
from .utils.time import humanize_seconds, maybe_make_aware
def populate_heap(self, event_t=event_t, heapify=heapq.heapify):
    """Populate the heap with the data contained in the schedule."""
    priority = 5
    self._heap = []
    for entry in self.schedule.values():
        is_due, next_call_delay = entry.is_due()
        self._heap.append(event_t(self._when(entry, 0 if is_due else next_call_delay) or 0, priority, entry))
    heapify(self._heap)