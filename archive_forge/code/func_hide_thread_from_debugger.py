import inspect
import os
import sys
def hide_thread_from_debugger(thread):
    """Disables tracing for the given thread if DEBUGPY_TRACE_DEBUGPY is not set.
    DEBUGPY_TRACE_DEBUGPY is used to debug debugpy with debugpy
    """
    if hide_debugpy_internals():
        thread.pydev_do_not_trace = True
        thread.is_pydev_daemon_thread = True