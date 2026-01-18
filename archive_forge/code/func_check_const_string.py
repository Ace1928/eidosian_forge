from ctypes import c_void_p, string_at
from django.contrib.gis.gdal.error import GDALException, SRSException, check_err
from django.contrib.gis.gdal.libgdal import lgdal
def check_const_string(result, func, cargs, offset=None, cpl=False):
    """
    Similar functionality to `check_string`, but does not free the pointer.
    """
    if offset:
        check_err(result, cpl=cpl)
        ptr = ptr_byref(cargs, offset)
        return ptr.value
    else:
        return result