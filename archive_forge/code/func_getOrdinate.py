from ctypes import byref, c_byte, c_double, c_uint
from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.libgeos import CS_PTR
from django.contrib.gis.shortcuts import numpy
def getOrdinate(self, dimension, index):
    """Return the value for the given dimension and index."""
    self._checkindex(index)
    self._checkdim(dimension)
    return capi.cs_getordinate(self.ptr, index, dimension, byref(c_double()))