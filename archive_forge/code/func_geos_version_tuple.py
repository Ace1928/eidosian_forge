import logging
import os
from ctypes import CDLL, CFUNCTYPE, POINTER, Structure, c_char_p
from ctypes.util import find_library
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import SimpleLazyObject, cached_property
from django.utils.version import get_version_tuple
def geos_version_tuple():
    """Return the GEOS version as a tuple (major, minor, subminor)."""
    return get_version_tuple(geos_version().decode())