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
def remove_quotes_from_args(args):
    if sys.platform == 'win32':
        new_args = []
        for x in args:
            if Path is not None and isinstance(x, Path):
                x = str(x)
            elif not isinstance(x, (bytes, str)):
                raise InvalidTypeInArgsException(str(type(x)))
            double_quote, two_double_quotes = _get_str_type_compatible(x, ['"', '""'])
            if x != two_double_quotes:
                if len(x) > 1 and x.startswith(double_quote) and x.endswith(double_quote):
                    x = x[1:-1]
            new_args.append(x)
        return new_args
    else:
        new_args = []
        for x in args:
            if Path is not None and isinstance(x, Path):
                x = x.as_posix()
            elif not isinstance(x, (bytes, str)):
                raise InvalidTypeInArgsException(str(type(x)))
            new_args.append(x)
        return new_args