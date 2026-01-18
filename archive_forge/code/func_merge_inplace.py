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
def merge_inplace(self, b):
    schedule = self.schedule
    A, B = (set(schedule), set(b))
    for key in A ^ B:
        schedule.pop(key, None)
    for key in B:
        entry = self.Entry(**dict(b[key], name=key, app=self.app))
        if schedule.get(key):
            schedule[key].update(entry)
        else:
            schedule[key] = entry