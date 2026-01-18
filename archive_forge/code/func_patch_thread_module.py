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
def patch_thread_module(thread_module):
    if getattr(thread_module, '_original_start_new_thread', None) is None:
        if thread_module is threading:
            if not hasattr(thread_module, '_start_new_thread'):
                return
            _original_start_new_thread = thread_module._original_start_new_thread = thread_module._start_new_thread
        else:
            _original_start_new_thread = thread_module._original_start_new_thread = thread_module.start_new_thread
    else:
        _original_start_new_thread = thread_module._original_start_new_thread

    class ClassWithPydevStartNewThread:

        def pydev_start_new_thread(self, function, args=(), kwargs={}):
            """
            We need to replace the original thread_module.start_new_thread with this function so that threads started
            through it and not through the threading module are properly traced.
            """
            return _original_start_new_thread(_UseNewThreadStartup(function, args, kwargs), ())
    pydev_start_new_thread = ClassWithPydevStartNewThread().pydev_start_new_thread
    try:
        if thread_module is threading:
            thread_module._start_new_thread = pydev_start_new_thread
        else:
            thread_module.start_new_thread = pydev_start_new_thread
            thread_module.start_new = pydev_start_new_thread
    except:
        pass