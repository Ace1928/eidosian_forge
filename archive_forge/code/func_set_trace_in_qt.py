from __future__ import nested_scopes
from _pydev_bundle._pydev_saved_modules import threading
import os
from _pydev_bundle import pydev_log
def set_trace_in_qt():
    from _pydevd_bundle.pydevd_comm import get_global_debugger
    py_db = get_global_debugger()
    if py_db is not None:
        threading.current_thread()
        py_db.enable_tracing()