import os as _os
import sys as _sys
import _thread
import functools
from time import monotonic as _time
from _weakrefset import WeakSet
from itertools import islice as _islice, count as _count
from _thread import stack_size
def gettrace():
    """Get the trace function as set by threading.settrace()."""
    return _trace_hook