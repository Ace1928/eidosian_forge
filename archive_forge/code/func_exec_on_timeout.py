from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_constants import thread_get_ident, IS_CPYTHON, NULL
import ctypes
import time
from _pydev_bundle import pydev_log
import weakref
from _pydevd_bundle.pydevd_utils import is_current_thread_main_thread
from _pydevd_bundle import pydevd_utils
def exec_on_timeout(self):
    kwargs = self.kwargs
    on_timeout = self.on_timeout
    if not self.disposed:
        self.disposed = True
        self.kwargs = None
        self.on_timeout = None
        try:
            if _DEBUG:
                pydev_log.critical('pydevd_timeout: Calling on timeout: %s with kwargs: %s', on_timeout, kwargs)
            on_timeout(**kwargs)
        except Exception:
            pydev_log.exception('pydevd_timeout: Exception on callback timeout.')