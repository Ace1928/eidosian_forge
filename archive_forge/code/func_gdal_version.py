import logging
import os
import re
from ctypes import CDLL, CFUNCTYPE, c_char_p, c_int
from ctypes.util import find_library
from django.contrib.gis.gdal.error import GDALException
from django.core.exceptions import ImproperlyConfigured
def gdal_version():
    """Return only the GDAL version number information."""
    return _version_info(b'RELEASE_NAME')