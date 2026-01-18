from _pydevd_bundle.pydevd_constants import get_current_thread_id, Null, ForkSafeLock
from pydevd_file_utils import get_abs_path_real_path_and_base_from_frame
from _pydev_bundle._pydev_saved_modules import thread, threading
import sys
from _pydev_bundle import pydev_log
def custom_frames_container_init():
    CustomFramesContainer.custom_frames_lock = ForkSafeLock()
    CustomFramesContainer.custom_frames = {}
    CustomFramesContainer._next_frame_id = 0
    CustomFramesContainer._py_db_command_thread_event = Null()