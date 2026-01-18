from ctypes import byref, c_double, c_int, c_void_p
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.prototypes import raster as capi
from django.contrib.gis.gdal.raster.base import GDALRasterBase
from django.contrib.gis.shortcuts import numpy
from django.utils.encoding import force_str
from .const import (
@nodata_value.setter
def nodata_value(self, value):
    """
        Set the nodata value for this band.
        """
    if value is None:
        capi.delete_band_nodata_value(self._ptr)
    elif not isinstance(value, (int, float)):
        raise ValueError('Nodata value must be numeric or None.')
    else:
        capi.set_band_nodata_value(self._ptr, value)
    self._flush()