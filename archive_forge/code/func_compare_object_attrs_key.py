from __future__ import nested_scopes
import traceback
import warnings
from _pydev_bundle import pydev_log
from _pydev_bundle._pydev_saved_modules import thread, threading
from _pydev_bundle import _pydev_saved_modules
import signal
import os
import ctypes
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from urllib.parse import quote  # @UnresolvedImport
import time
import inspect
import sys
from _pydevd_bundle.pydevd_constants import USE_CUSTOM_SYS_CURRENT_FRAMES, IS_PYPY, SUPPORT_GEVENT, \
def compare_object_attrs_key(x):
    if GENERATED_LEN_ATTR_NAME == x:
        as_number = to_number(x)
        if as_number is None:
            as_number = 99999999
        return (1, as_number)
    else:
        return (-1, to_string(x))