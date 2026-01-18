from _pydevd_bundle.pydevd_constants import (STATE_RUN, PYTHON_SUSPEND, SUPPORT_GEVENT, ForkSafeLock,
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_frame import PyDBFrame
def set_additional_thread_info(thread):
    try:
        additional_info = thread.additional_info
        if additional_info is None:
            raise AttributeError()
    except:
        with _set_additional_thread_info_lock:
            additional_info = getattr(thread, 'additional_info', None)
            if additional_info is None:
                additional_info = PyDBAdditionalThreadInfo()
            thread.additional_info = additional_info
    return additional_info