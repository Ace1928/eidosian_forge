import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def stop_here(self, frame):
    """Return True if frame is below the starting frame in the stack."""
    if self.skip and self.is_skipped_module(frame.f_globals.get('__name__')):
        return False
    if frame is self.stopframe:
        if self.stoplineno == -1:
            return False
        return frame.f_lineno >= self.stoplineno
    if not self.stopframe:
        return True
    return False