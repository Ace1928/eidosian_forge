from ctypes import c_void_p, string_at
from django.contrib.gis.gdal.error import GDALException, SRSException, check_err
from django.contrib.gis.gdal.libgdal import lgdal
def check_envelope(result, func, cargs, offset=-1):
    """Check a function that returns an OGR Envelope by reference."""
    return ptr_byref(cargs, offset)