from __future__ import division
import os
import sys
from math import sqrt
import functools
import collections
import time
import threading
import itertools
from uuid import uuid4
from numbers import Integral
import warnings
import queue
import weakref
from contextlib import nullcontext
from multiprocessing import TimeoutError
from ._multiprocessing_helpers import mp
from .logger import Logger, short_format_time
from .disk import memstr_to_bytes
from ._parallel_backends import (FallbackToBackend, MultiprocessingBackend,
from ._utils import eval_expr, _Sentinel
from ._parallel_backends import AutoBatchingMixin  # noqa
from ._parallel_backends import ParallelBackendBase  # noqa
def print_progress(self):
    """Display the process of the parallel execution only a fraction
           of time, controlled by self.verbose.
        """
    if not self.verbose:
        return
    elapsed_time = time.time() - self._start_time
    if self._is_completed():
        self._print(f'Done {self.n_completed_tasks:3d} out of {self.n_completed_tasks:3d} | elapsed: {short_format_time(elapsed_time)} finished')
        return
    elif self._original_iterator is not None:
        if _verbosity_filter(self.n_dispatched_batches, self.verbose):
            return
        self._print(f'Done {self.n_completed_tasks:3d} tasks      | elapsed: {short_format_time(elapsed_time)}')
    else:
        index = self.n_completed_tasks
        total_tasks = self.n_dispatched_tasks
        if not index == 0:
            cursor = total_tasks - index + 1 - self._pre_dispatch_amount
            frequency = total_tasks // self.verbose + 1
            is_last_item = index + 1 == total_tasks
            if is_last_item or cursor % frequency:
                return
        remaining_time = elapsed_time / index * (self.n_dispatched_tasks - index * 1.0)
        self._print(f'Done {index:3d} out of {total_tasks:3d} | elapsed: {short_format_time(elapsed_time)} remaining: {short_format_time(remaining_time)}')