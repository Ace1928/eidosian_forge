import inspect
import os
import sys
def hide_debugpy_internals():
    """Returns True if the caller should hide something from debugpy."""
    return 'DEBUGPY_TRACE_DEBUGPY' not in os.environ