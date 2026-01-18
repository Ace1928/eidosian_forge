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
@geotransform.setter
def geotransform(self, values):
    """Set the geotransform for the data source."""
    if len(values) != 6 or not all((isinstance(x, (int, float)) for x in values)):
        raise ValueError('Geotransform must consist of 6 numeric values.')
    values = (c_double * 6)(*values)
    capi.set_ds_geotransform(self._ptr, byref(values))
    self._flush()