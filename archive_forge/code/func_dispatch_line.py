import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def dispatch_line(self, frame):
    """Invoke user function and return trace function for line event.

        If the debugger stops on the current line, invoke
        self.user_line(). Raise BdbQuit if self.quitting is set.
        Return self.trace_dispatch to continue tracing in this scope.
        """
    if self.stop_here(frame) or self.break_here(frame):
        self.user_line(frame)
        if self.quitting:
            raise BdbQuit
    return self.trace_dispatch