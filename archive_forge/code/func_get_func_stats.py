import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def get_func_stats(tag=None, ctx_id=None, filter=None, filter_callback=None):
    """
    Gets the function profiler results with given filters and returns an iterable.

    filter: is here mainly for backward compat. we will not document it anymore.
    tag, ctx_id: select given tag and ctx_id related stats in C side.
    filter_callback: we could do it like: get_func_stats().filter(). The problem
    with this approach is YFuncStats has an internal list which complicates:
        - delete() operation because list deletions are O(n)
        - sort() and pop() operations currently work on sorted list and they hold the
          list as sorted.
    To preserve above behaviour and have a delete() method, we can use an OrderedDict()
    maybe, but simply that is not worth the effort for an extra filter() call. Maybe
    in the future.
    """
    if not filter:
        filter = {}
    if tag:
        filter['tag'] = tag
    if ctx_id:
        filter['ctx_id'] = ctx_id
    _yappi._pause()
    try:
        stats = YFuncStats().get(filter=filter, filter_callback=filter_callback)
    finally:
        _yappi._resume()
    return stats