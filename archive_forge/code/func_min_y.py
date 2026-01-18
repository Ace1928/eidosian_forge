from ctypes import Structure, c_double
from django.contrib.gis.gdal.error import GDALException
@property
def min_y(self):
    """Return the value of the minimum Y coordinate."""
    return self._envelope.MinY