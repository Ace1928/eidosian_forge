from ctypes import c_void_p, string_at
from django.contrib.gis.gdal.error import GDALException, SRSException, check_err
from django.contrib.gis.gdal.libgdal import lgdal
def check_srs(result, func, cargs):
    if isinstance(result, int):
        result = c_void_p(result)
    if not result:
        raise SRSException('Invalid spatial reference pointer returned from "%s".' % func.__name__)
    return result