import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def set_return(self, frame):
    """Stop when returning from the given frame."""
    if frame.f_code.co_flags & GENERATOR_AND_COROUTINE_FLAGS:
        self._set_stopinfo(frame, None, -1)
    else:
        self._set_stopinfo(frame.f_back, frame)