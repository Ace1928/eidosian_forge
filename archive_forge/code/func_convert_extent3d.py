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
def convert_extent3d(self, box3d):
    """
        Return a 6-tuple extent for the `Extent3D` aggregate by converting
        the 3d bounding-box text returned by PostGIS (`box3d` argument), for
        example: "BOX3D(-90.0 30.0 1, -85.0 40.0 2)".
        """
    if box3d is None:
        return None
    ll, ur = box3d[6:-1].split(',')
    xmin, ymin, zmin = map(float, ll.split())
    xmax, ymax, zmax = map(float, ur.split())
    return (xmin, ymin, zmin, xmax, ymax, zmax)