from collections import defaultdict, namedtuple
from django.contrib.gis import forms, gdal
from django.contrib.gis.db.models.proxy import SpatialProxy
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.geos import (
from django.core.exceptions import ImproperlyConfigured
from django.db.models import Field
from django.utils.translation import gettext_lazy as _
def get_raster_prep_value(self, value, is_candidate):
    """
        Return a GDALRaster if conversion is successful, otherwise return None.
        """
    if isinstance(value, gdal.GDALRaster):
        return value
    elif is_candidate:
        try:
            return gdal.GDALRaster(value)
        except GDALException:
            pass
    elif isinstance(value, dict):
        try:
            return gdal.GDALRaster(value)
        except GDALException:
            raise ValueError("Couldn't create spatial object from lookup value '%s'." % value)