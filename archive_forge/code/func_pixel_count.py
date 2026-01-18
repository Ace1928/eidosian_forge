from ctypes import byref, c_double, c_int, c_void_p
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.prototypes import raster as capi
from django.contrib.gis.gdal.raster.base import GDALRasterBase
from django.contrib.gis.shortcuts import numpy
from django.utils.encoding import force_str
from .const import (
@property
def pixel_count(self):
    """
        Return the total number of pixels in this band.
        """
    return self.width * self.height