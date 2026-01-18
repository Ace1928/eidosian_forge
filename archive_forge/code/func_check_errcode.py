from ctypes import c_void_p, string_at
from django.contrib.gis.gdal.error import GDALException, SRSException, check_err
from django.contrib.gis.gdal.libgdal import lgdal
def check_errcode(result, func, cargs, cpl=False):
    """
    Check the error code returned (c_int).
    """
    check_err(result, cpl=cpl)