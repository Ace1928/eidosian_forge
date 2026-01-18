from django.contrib.gis.db import models
from django.contrib.gis.db.backends.base.operations import BaseSpatialOperations
from django.contrib.gis.db.backends.spatialite.adapter import SpatiaLiteAdapter
from django.contrib.gis.db.backends.utils import SpatialOperator
from django.contrib.gis.geos.geometry import GEOSGeometry, GEOSGeometryBase
from django.contrib.gis.geos.prototypes.io import wkb_r
from django.contrib.gis.measure import Distance
from django.core.exceptions import ImproperlyConfigured
from django.db.backends.sqlite3.operations import DatabaseOperations
from django.utils.functional import cached_property
from django.utils.version import get_version_tuple
def spatialite_version_tuple(self):
    """
        Return the SpatiaLite version as a tuple (version string, major,
        minor, subminor).
        """
    version = self.spatialite_version()
    return (version,) + get_version_tuple(version)