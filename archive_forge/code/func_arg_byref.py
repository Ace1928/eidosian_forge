from ctypes import c_void_p, string_at
from django.contrib.gis.gdal.error import GDALException, SRSException, check_err
from django.contrib.gis.gdal.libgdal import lgdal
def arg_byref(args, offset=-1):
    """Return the pointer argument's by-reference value."""
    return args[offset]._obj.value