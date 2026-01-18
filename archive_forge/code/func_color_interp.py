from ctypes import byref, c_double, c_int, c_void_p
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.prototypes import raster as capi
from django.contrib.gis.gdal.raster.base import GDALRasterBase
from django.contrib.gis.shortcuts import numpy
from django.utils.encoding import force_str
from .const import (
def color_interp(self, as_string=False):
    """Return the GDAL color interpretation for this band."""
    color = capi.get_band_color_interp(self._ptr)
    if as_string:
        color = GDAL_COLOR_TYPES[color]
    return color