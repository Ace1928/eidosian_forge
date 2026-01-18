from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_constants import thread_get_ident, IS_CPYTHON, NULL
import ctypes
import time
from _pydev_bundle import pydev_log
import weakref
from _pydevd_bundle.pydevd_utils import is_current_thread_main_thread
from _pydevd_bundle import pydevd_utils
def process_handles(self):
    """
        :return int:
            Returns the time we should be waiting for to process the next event properly.
        """
    with self._lock:
        if _DEBUG:
            pydev_log.critical('pydevd_timeout: Processing handles')
        self._event.clear()
        handles = self._handles
        new_handles = self._handles = []
        curtime = time.time()
        min_handle_timeout = None
        for handle in handles:
            if curtime < handle.abs_timeout and (not handle.disposed):
                if _DEBUG:
                    pydev_log.critical('pydevd_timeout: Handle NOT processed: %s', handle)
                new_handles.append(handle)
                if min_handle_timeout is None:
                    min_handle_timeout = handle.abs_timeout
                elif handle.abs_timeout < min_handle_timeout:
                    min_handle_timeout = handle.abs_timeout
            else:
                if _DEBUG:
                    pydev_log.critical('pydevd_timeout: Handle processed: %s', handle)
                handle.exec_on_timeout()
        if min_handle_timeout is None:
            return None
        else:
            timeout = min_handle_timeout - curtime
            if timeout <= 0:
                pydev_log.critical('pydevd_timeout: Expected timeout to be > 0. Found: %s', timeout)
            return timeout