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
def create_warn_multiproc(original_name):

    def new_warn_multiproc(*args, **kwargs):
        import os
        warn_multiproc()
        return getattr(os, original_name)(*args, **kwargs)
    return new_warn_multiproc