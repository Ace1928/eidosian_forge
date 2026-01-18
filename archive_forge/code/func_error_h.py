import logging
import os
from ctypes import CDLL, CFUNCTYPE, POINTER, Structure, c_char_p
from ctypes.util import find_library
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import SimpleLazyObject, cached_property
from django.utils.version import get_version_tuple
def error_h(fmt, lst):
    fmt, lst = (fmt.decode(), lst.decode())
    try:
        err_msg = fmt % lst
    except TypeError:
        err_msg = fmt
    logger.error('GEOS_ERROR: %s\n', err_msg)