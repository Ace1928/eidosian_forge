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
def get_thread_id_for_variable_reference(self, variable_reference):
    """
        We can't evaluate variable references values on any thread, only in the suspended
        thread (the main reason for this is that in UI frameworks inspecting a UI object
        from a different thread can potentially crash the application).

        :param int variable_reference:
            The variable reference (can be either a frame id or a reference to a previously
            gotten variable).

        :return str:
            The thread id for the thread to be used to inspect the given variable reference or
            None if the thread was already resumed.
        """
    frames_tracker = self._get_tracker_for_variable_reference(variable_reference)
    if frames_tracker is not None:
        return frames_tracker.get_main_thread_id()
    return None