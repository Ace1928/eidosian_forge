import os as _os
import sys as _sys
import _thread
import functools
from time import monotonic as _time
from _weakrefset import WeakSet
from itertools import islice as _islice, count as _count
from _thread import stack_size
@property
def native_id(self):
    """Native integral thread ID of this thread, or None if it has not been started.

            This is a non-negative integer. See the get_native_id() function.
            This represents the Thread ID as reported by the kernel.

            """
    assert self._initialized, 'Thread.__init__() not called'
    return self._native_id