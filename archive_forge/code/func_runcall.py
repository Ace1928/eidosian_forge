import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def runcall(self, func, /, *args, **kwds):
    """Debug a single function call.

        Return the result of the function call.
        """
    self.reset()
    sys.settrace(self.trace_dispatch)
    res = None
    try:
        res = func(*args, **kwds)
    except BdbQuit:
        pass
    finally:
        self.quitting = True
        sys.settrace(None)
    return res