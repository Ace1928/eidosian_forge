import os
import re
import sys
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import get_global_debugger, IS_WINDOWS, IS_JYTHON, get_current_thread_id, \
from _pydev_bundle import pydev_log
from contextlib import contextmanager
from _pydevd_bundle import pydevd_constants, pydevd_defaults
from _pydevd_bundle.pydevd_defaults import PydevdCustomization
import ast
def new_fork():
    is_new_python_process = True
    frame = sys._getframe()
    apply_arg_patch = _get_apply_arg_patching()
    is_subprocess_fork = False
    while frame is not None:
        if frame.f_code.co_name == '_execute_child' and 'subprocess' in frame.f_code.co_filename:
            is_subprocess_fork = True
            executable = frame.f_locals.get('executable')
            if executable is not None:
                is_new_python_process = False
                if is_python(executable):
                    is_new_python_process = True
            break
        frame = frame.f_back
    frame = None
    protocol = pydevd_constants.get_protocol()
    debug_mode = PydevdCustomization.DEBUG_MODE
    child_process = getattr(os, original_name)()
    if not child_process:
        if is_new_python_process:
            PydevdCustomization.DEFAULT_PROTOCOL = protocol
            PydevdCustomization.DEBUG_MODE = debug_mode
            _on_forked_process(setup_tracing=apply_arg_patch and (not is_subprocess_fork))
        else:
            set_global_debugger(None)
    elif is_new_python_process:
        send_process_created_message()
    return child_process