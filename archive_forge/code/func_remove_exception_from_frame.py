from _pydevd_bundle.pydevd_constants import EXCEPTION_TYPE_USER_UNHANDLED, EXCEPTION_TYPE_UNHANDLED, \
from _pydev_bundle import pydev_log
import itertools
from typing import Any, Dict
def remove_exception_from_frame(frame):
    frame.f_locals.pop('__exception__', None)