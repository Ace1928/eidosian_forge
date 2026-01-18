from django.contrib.gis.db import models
from django.contrib.gis.db.backends.base.adapter import WKTAdapter
from django.contrib.gis.db.backends.base.operations import BaseSpatialOperations
from django.contrib.gis.db.backends.utils import SpatialOperator
from django.contrib.gis.geos.geometry import GEOSGeometryBase
from django.contrib.gis.geos.prototypes.io import wkb_r
from django.contrib.gis.measure import Distance
from django.db.backends.mysql.operations import DatabaseOperations
from django.utils.functional import cached_property
@cached_property
def gis_operators(self):
    operators = {'bbcontains': SpatialOperator(func='MBRContains'), 'bboverlaps': SpatialOperator(func='MBROverlaps'), 'contained': SpatialOperator(func='MBRWithin'), 'contains': SpatialOperator(func='ST_Contains'), 'crosses': SpatialOperator(func='ST_Crosses'), 'disjoint': SpatialOperator(func='ST_Disjoint'), 'equals': SpatialOperator(func='ST_Equals'), 'exact': SpatialOperator(func='ST_Equals'), 'intersects': SpatialOperator(func='ST_Intersects'), 'overlaps': SpatialOperator(func='ST_Overlaps'), 'same_as': SpatialOperator(func='ST_Equals'), 'touches': SpatialOperator(func='ST_Touches'), 'within': SpatialOperator(func='ST_Within')}
    if self.connection.mysql_is_mariadb:
        operators['relate'] = SpatialOperator(func='ST_Relate')
    return operators