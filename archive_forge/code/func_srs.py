import json
import os
import sys
import uuid
from ctypes import (
from pathlib import Path
from django.contrib.gis.gdal.driver import Driver
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.prototypes import raster as capi
from django.contrib.gis.gdal.raster.band import BandList
from django.contrib.gis.gdal.raster.base import GDALRasterBase
from django.contrib.gis.gdal.raster.const import (
from django.contrib.gis.gdal.srs import SpatialReference, SRSException
from django.contrib.gis.geometry import json_regex
from django.utils.encoding import force_bytes, force_str
from django.utils.functional import cached_property
@srs.setter
def srs(self, value):
    """
        Set the spatial reference used in this GDALRaster. The input can be
        a SpatialReference or any parameter accepted by the SpatialReference
        constructor.
        """
    if isinstance(value, SpatialReference):
        srs = value
    elif isinstance(value, (int, str)):
        srs = SpatialReference(value)
    else:
        raise ValueError('Could not create a SpatialReference from input.')
    capi.set_ds_projection_ref(self._ptr, srs.wkt.encode())
    self._flush()