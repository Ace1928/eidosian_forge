from ctypes import c_void_p, string_at
from django.contrib.gis.gdal.error import GDALException, SRSException, check_err
from django.contrib.gis.gdal.libgdal import lgdal
def check_arg_errcode(result, func, cargs, cpl=False):
    """
    The error code is returned in the last argument, by reference.
    Check its value with `check_err` before returning the result.
    """
    check_err(arg_byref(cargs), cpl=cpl)
    return result