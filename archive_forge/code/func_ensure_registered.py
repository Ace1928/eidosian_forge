from ctypes import c_void_p
from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.prototypes import ds as vcapi
from django.contrib.gis.gdal.prototypes import raster as rcapi
from django.utils.encoding import force_bytes, force_str
@classmethod
def ensure_registered(cls):
    """
        Attempt to register all the data source drivers.
        """
    if not vcapi.get_driver_count():
        vcapi.register_all()
    if not rcapi.get_driver_count():
        rcapi.register_all()