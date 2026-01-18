from _pydev_bundle._pydev_saved_modules import threading
from _pydev_bundle import _pydev_saved_modules
from _pydevd_bundle.pydevd_utils import notify_about_gevent_if_needed
import weakref
from _pydevd_bundle.pydevd_constants import IS_JYTHON, IS_IRONPYTHON, \
from _pydev_bundle.pydev_log import exception as pydev_log_exception
import sys
from _pydev_bundle import pydev_log
import pydevd_tracing
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
def mark_as_pydevd_daemon_thread(thread):
    if not IS_JYTHON and (not IS_IRONPYTHON) and PYDEVD_APPLY_PATCHING_TO_HIDE_PYDEVD_THREADS:
        global _patched_threading_to_hide_pydevd_threads
        if not _patched_threading_to_hide_pydevd_threads:
            _patched_threading_to_hide_pydevd_threads = True
            try:
                _patch_threading_to_hide_pydevd_threads()
            except:
                pydev_log.exception('Error applying patching to hide pydevd threads.')
    thread.pydev_do_not_trace = True
    thread.is_pydev_daemon_thread = True
    thread.daemon = True