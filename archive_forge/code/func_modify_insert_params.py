import re
from django.contrib.gis.db import models
from django.contrib.gis.db.backends.base.operations import BaseSpatialOperations
from django.contrib.gis.db.backends.oracle.adapter import OracleSpatialAdapter
from django.contrib.gis.db.backends.utils import SpatialOperator
from django.contrib.gis.geos.geometry import GEOSGeometry, GEOSGeometryBase
from django.contrib.gis.geos.prototypes.io import wkb_r
from django.contrib.gis.measure import Distance
from django.db.backends.oracle.operations import DatabaseOperations
def modify_insert_params(self, placeholder, params):
    """Drop out insert parameters for NULL placeholder. Needed for Oracle Spatial
        backend due to #10888.
        """
    if placeholder == 'NULL':
        return []
    return super().modify_insert_params(placeholder, params)