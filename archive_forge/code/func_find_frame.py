from contextlib import contextmanager
import sys
from _pydevd_bundle.pydevd_constants import get_frame, RETURN_VALUES_DICT, \
from _pydevd_bundle.pydevd_xml import get_variable_details, get_type
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle.pydevd_resolver import sorted_attributes_key, TOO_LARGE_ATTR, get_var_scope
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_vars
from _pydev_bundle.pydev_imports import Exec
from _pydevd_bundle.pydevd_frame_utils import FramesList
from _pydevd_bundle.pydevd_utils import ScopeRequest, DAPGrouper, Timer
from typing import Optional
def find_frame(self, thread_id, frame_id):
    try:
        if frame_id == '*':
            return get_frame()
        frame_id = int(frame_id)
        fake_frames = self._thread_id_to_fake_frames.get(thread_id)
        if fake_frames is not None:
            frame = fake_frames.get(frame_id)
            if frame is not None:
                return frame
        frames_tracker = self._thread_id_to_tracker.get(thread_id)
        if frames_tracker is not None:
            frame = frames_tracker.find_frame(thread_id, frame_id)
            if frame is not None:
                return frame
        return None
    except:
        pydev_log.exception()
        return None