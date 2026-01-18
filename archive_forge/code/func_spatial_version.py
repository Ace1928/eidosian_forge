import re
from django.conf import settings
from django.contrib.gis.db.backends.base.operations import BaseSpatialOperations
from django.contrib.gis.db.backends.utils import SpatialOperator
from django.contrib.gis.db.models import GeometryField, RasterField
from django.contrib.gis.gdal import GDALRaster
from django.contrib.gis.geos.geometry import GEOSGeometryBase
from django.contrib.gis.geos.prototypes.io import wkb_r
from django.contrib.gis.measure import Distance
from django.core.exceptions import ImproperlyConfigured
from django.db import NotSupportedError, ProgrammingError
from django.db.backends.postgresql.operations import DatabaseOperations
from django.db.backends.postgresql.psycopg_any import is_psycopg3
from django.db.models import Func, Value
from django.utils.functional import cached_property
from django.utils.version import get_version_tuple
from .adapter import PostGISAdapter
from .models import PostGISGeometryColumns, PostGISSpatialRefSys
from .pgraster import from_pgraster
@cached_property
def spatial_version(self):
    """Determine the version of the PostGIS library."""
    if hasattr(settings, 'POSTGIS_VERSION'):
        version = settings.POSTGIS_VERSION
    else:
        self._get_postgis_func('version')
        try:
            vtup = self.postgis_version_tuple()
        except ProgrammingError:
            raise ImproperlyConfigured('Cannot determine PostGIS version for database "%s" using command "SELECT postgis_lib_version()". GeoDjango requires at least PostGIS version 2.5. Was the database created from a spatial database template?' % self.connection.settings_dict['NAME'])
        version = vtup[1:]
    return version