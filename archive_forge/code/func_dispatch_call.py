import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def dispatch_call(self, frame, arg):
    """Invoke user function and return trace function for call event.

        If the debugger stops on this function call, invoke
        self.user_call(). Raise BdbQuit if self.quitting is set.
        Return self.trace_dispatch to continue tracing in this scope.
        """
    if self.botframe is None:
        self.botframe = frame.f_back
        return self.trace_dispatch
    if not (self.stop_here(frame) or self.break_anywhere(frame)):
        return
    if self.stopframe and frame.f_code.co_flags & GENERATOR_AND_COROUTINE_FLAGS:
        return self.trace_dispatch
    self.user_call(frame, arg)
    if self.quitting:
        raise BdbQuit
    return self.trace_dispatch