from ctypes import byref, c_char_p, c_int
from enum import IntEnum
from types import NoneType
from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.error import SRSException
from django.contrib.gis.gdal.libgdal import GDAL_VERSION
from django.contrib.gis.gdal.prototypes import srs as capi
from django.utils.encoding import force_bytes, force_str
def import_wkt(self, wkt):
    """Import the Spatial Reference from OGC WKT (string)"""
    capi.from_wkt(self.ptr, byref(c_char_p(force_bytes(wkt))))