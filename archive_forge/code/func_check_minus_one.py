from ctypes import c_void_p, string_at
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.libgeos import GEOSFuncFactory
def check_minus_one(result, func, cargs):
    """Error checking on routines that should not return -1."""
    if result == -1:
        raise GEOSException('Error encountered in GEOS C function "%s".' % func.__name__)
    else:
        return result