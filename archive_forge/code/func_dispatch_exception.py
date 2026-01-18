import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def dispatch_exception(self, frame, arg):
    """Invoke user function and return trace function for exception event.

        If the debugger stops on this exception, invoke
        self.user_exception(). Raise BdbQuit if self.quitting is set.
        Return self.trace_dispatch to continue tracing in this scope.
        """
    if self.stop_here(frame):
        if not (frame.f_code.co_flags & GENERATOR_AND_COROUTINE_FLAGS and arg[0] is StopIteration and (arg[2] is None)):
            self.user_exception(frame, arg)
            if self.quitting:
                raise BdbQuit
    elif self.stopframe and frame is not self.stopframe and self.stopframe.f_code.co_flags & GENERATOR_AND_COROUTINE_FLAGS and (arg[0] in (StopIteration, GeneratorExit)):
        self.user_exception(frame, arg)
        if self.quitting:
            raise BdbQuit
    return self.trace_dispatch