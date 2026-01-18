from _pydevd_bundle.pydevd_constants import get_current_thread_id, Null, ForkSafeLock
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame
from _pydev_bundle._pydev_saved_modules import thread, threading
import sys
from _pydev_bundle import pydev_log
def remove_custom_frame(frame_custom_thread_id):
    with CustomFramesContainer.custom_frames_lock:
        if DEBUG:
            sys.stderr.write('remove_custom_frame: %s\n' % frame_custom_thread_id)
        CustomFramesContainer.custom_frames.pop(frame_custom_thread_id, None)
        CustomFramesContainer._py_db_command_thread_event.set()