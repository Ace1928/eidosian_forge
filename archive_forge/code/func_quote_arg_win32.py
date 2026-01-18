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
def quote_arg_win32(arg):
    fix_type = lambda x: _get_str_type_compatible(arg, x)
    if arg and (not set(arg).intersection(fix_type(' "\t\n\x0b'))):
        return arg
    arg = re.sub(fix_type('(\\\\*)\\"'), fix_type('\\1\\1\\\\"'), arg)
    arg = re.sub(fix_type('(\\\\*)$'), fix_type('\\1\\1'), arg)
    return fix_type('"') + arg + fix_type('"')