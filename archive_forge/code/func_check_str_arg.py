from ctypes import c_void_p, string_at
from django.contrib.gis.gdal.error import GDALException, SRSException, check_err
from django.contrib.gis.gdal.libgdal import lgdal
def check_str_arg(result, func, cargs):
    """
    This is for the OSRGet[Angular|Linear]Units functions, which
    require that the returned string pointer not be freed.  This
    returns both the double and string values.
    """
    dbl = result
    ptr = cargs[-1]._obj
    return (dbl, ptr.value.decode())