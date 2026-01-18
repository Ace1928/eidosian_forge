from ctypes import Structure, c_double
from django.contrib.gis.gdal.error import GDALException
@property
def max_y(self):
    """Return the value of the maximum Y coordinate."""
    return self._envelope.MaxY