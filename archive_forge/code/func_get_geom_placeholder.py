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
def get_geom_placeholder(self, f, value, compiler):
    """
        Provide a proper substitution value for Geometries or rasters that are
        not in the SRID of the field. Specifically, this routine will
        substitute in the ST_Transform() function call.
        """
    transform_func = self.spatial_function_name('Transform')
    if hasattr(value, 'as_sql'):
        if value.field.srid == f.srid:
            placeholder = '%s'
        else:
            placeholder = '%s(%%s, %s)' % (transform_func, f.srid)
        return placeholder
    if value is None:
        value_srid = None
    else:
        value_srid = value.srid
    if value_srid is None or value_srid == f.srid:
        placeholder = '%s'
    else:
        placeholder = '%s(%%s, %s)' % (transform_func, f.srid)
    return placeholder