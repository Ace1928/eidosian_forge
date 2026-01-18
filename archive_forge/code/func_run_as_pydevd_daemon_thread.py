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
def run_as_pydevd_daemon_thread(py_db, func, *args, **kwargs):
    """
    Runs a function as a pydevd daemon thread (without any tracing in place).
    """
    t = PyDBDaemonThread(py_db, target_and_args=(func, args, kwargs))
    t.name = '%s (pydevd daemon thread)' % (func.__name__,)
    t.start()
    return t