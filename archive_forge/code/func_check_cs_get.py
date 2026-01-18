from ctypes import POINTER, c_byte, c_double, c_int, c_uint
from django.contrib.gis.geos.libgeos import CS_PTR, GEOM_PTR, GEOSFuncFactory
from django.contrib.gis.geos.prototypes.errcheck import GEOSException, last_arg_byref
def check_cs_get(result, func, cargs):
    """Check the coordinate sequence retrieval."""
    check_cs_op(result, func, cargs)
    return last_arg_byref(cargs)