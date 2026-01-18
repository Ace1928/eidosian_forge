from ctypes import POINTER, c_byte, c_double, c_int, c_uint
from django.contrib.gis.geos.libgeos import CS_PTR, GEOM_PTR, GEOSFuncFactory
from django.contrib.gis.geos.prototypes.errcheck import GEOSException, last_arg_byref
def check_cs_op(result, func, cargs):
    """Check the status code of a coordinate sequence operation."""
    if result == 0:
        raise GEOSException('Could not set value on coordinate sequence')
    else:
        return result