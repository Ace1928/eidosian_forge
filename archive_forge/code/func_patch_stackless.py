from __future__ import nested_scopes
import weakref
import sys
from _pydevd_bundle.pydevd_comm import get_global_debugger
from _pydevd_bundle.pydevd_constants import call_only_once
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_custom_frames import update_custom_frame, remove_custom_frame, add_custom_frame
import stackless  # @UnresolvedImport
from _pydev_bundle import pydev_log
def patch_stackless():
    """
    This function should be called to patch the stackless module so that new tasklets are properly tracked in the
    debugger.
    """
    global _application_set_schedule_callback
    _application_set_schedule_callback = stackless.set_schedule_callback(_schedule_callback)

    def set_schedule_callback(callable):
        global _application_set_schedule_callback
        old = _application_set_schedule_callback
        _application_set_schedule_callback = callable
        return old

    def get_schedule_callback():
        global _application_set_schedule_callback
        return _application_set_schedule_callback
    set_schedule_callback.__doc__ = stackless.set_schedule_callback.__doc__
    if hasattr(stackless, 'get_schedule_callback'):
        get_schedule_callback.__doc__ = stackless.get_schedule_callback.__doc__
    stackless.set_schedule_callback = set_schedule_callback
    stackless.get_schedule_callback = get_schedule_callback
    if not hasattr(stackless.tasklet, 'trace_function'):
        __call__.__doc__ = stackless.tasklet.__call__.__doc__
        stackless.tasklet.__call__ = __call__
        setup.__doc__ = stackless.tasklet.setup.__doc__
        stackless.tasklet.setup = setup
        run.__doc__ = stackless.run.__doc__
        stackless.run = run