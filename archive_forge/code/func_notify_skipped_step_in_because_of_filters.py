from _pydev_bundle.pydev_is_thread_alive import is_thread_alive
from _pydev_bundle.pydev_log import exception as pydev_log_exception
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import (get_current_thread_id, NO_FTRACE,
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame, NORM_PATHS_AND_BASE_CONTAINER
from _pydevd_bundle.pydevd_frame import PyDBFrame, is_unhandled_exception
def notify_skipped_step_in_because_of_filters(py_db, frame):
    global _global_notify_skipped_step_in
    with _global_notify_skipped_step_in_lock:
        if _global_notify_skipped_step_in:
            return
        _global_notify_skipped_step_in = True
        py_db.notify_skipped_step_in_because_of_filters(frame)