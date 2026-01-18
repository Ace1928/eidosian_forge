from ctypes import c_void_p, string_at
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.libgeos import GEOSFuncFactory
def check_predicate(result, func, cargs):
    """Error checking for unary/binary predicate functions."""
    if result == 1:
        return True
    elif result == 0:
        return False
    else:
        raise GEOSException('Error encountered on GEOS C predicate function "%s".' % func.__name__)