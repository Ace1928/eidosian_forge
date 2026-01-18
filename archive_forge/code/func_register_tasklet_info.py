from __future__ import nested_scopes
import weakref
import sys
from _pydevd_bundle.pydevd_comm import get_global_debugger
from _pydevd_bundle.pydevd_constants import call_only_once
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_custom_frames import update_custom_frame, remove_custom_frame, add_custom_frame
import stackless  # @UnresolvedImport
from _pydev_bundle import pydev_log
def register_tasklet_info(tasklet):
    r = weakref.ref(tasklet)
    info = _weak_tasklet_registered_to_info.get(r)
    if info is None:
        info = _weak_tasklet_registered_to_info[r] = _TaskletInfo(r, tasklet)
    return info