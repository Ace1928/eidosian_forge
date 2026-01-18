from collections.abc import Sequence, Iterable
from functools import total_ordering
import fnmatch
import linecache
import os.path
import pickle
from _tracemalloc import *
from _tracemalloc import _get_object_traceback, _get_traces
def take_snapshot():
    """
    Take a snapshot of traces of memory blocks allocated by Python.
    """
    if not is_tracing():
        raise RuntimeError('the tracemalloc module must be tracing memory allocations to take a snapshot')
    traces = _get_traces()
    traceback_limit = get_traceback_limit()
    return Snapshot(traces, traceback_limit)