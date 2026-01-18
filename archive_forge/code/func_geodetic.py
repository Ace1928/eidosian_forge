from collections import defaultdict, namedtuple
from django.contrib.gis import forms, gdal
from django.contrib.gis.db.models.proxy import SpatialProxy
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.geos import (
from django.core.exceptions import ImproperlyConfigured
from django.db.models import Field
from django.utils.translation import gettext_lazy as _
def geodetic(self, connection):
    """
        Return true if this field's SRID corresponds with a coordinate
        system that uses non-projected units (e.g., latitude/longitude).
        """
    return get_srid_info(self.srid, connection).geodetic