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
def patch_arg_str_win(arg_str):
    args = str_to_args_windows(arg_str)
    if not args or not is_python(args[0]):
        return arg_str
    arg_str = ' '.join(patch_args(args))
    pydev_log.debug('New args: %s', arg_str)
    return arg_str