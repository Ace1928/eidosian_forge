from ctypes import byref, c_byte, c_double, c_uint
from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.libgeos import CS_PTR
from django.contrib.gis.shortcuts import numpy
@property
def is_counterclockwise(self):
    """Return whether this coordinate sequence is counterclockwise."""
    ret = c_byte()
    if not capi.cs_is_ccw(self.ptr, byref(ret)):
        raise GEOSException('Error encountered in GEOS C function "%s".' % capi.cs_is_ccw.func_name)
    return ret.value == 1