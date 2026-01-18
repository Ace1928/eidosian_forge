import re
from django.contrib.gis.db import models
from django.contrib.gis.db.backends.base.operations import BaseSpatialOperations
from django.contrib.gis.db.backends.oracle.adapter import OracleSpatialAdapter
from django.contrib.gis.db.backends.utils import SpatialOperator
from django.contrib.gis.geos.geometry import GEOSGeometry, GEOSGeometryBase
from django.contrib.gis.geos.prototypes.io import wkb_r
from django.contrib.gis.measure import Distance
from django.db.backends.oracle.operations import DatabaseOperations
def check_relate_argument(self, arg):
    masks = 'TOUCH|OVERLAPBDYDISJOINT|OVERLAPBDYINTERSECT|EQUAL|INSIDE|COVEREDBY|CONTAINS|COVERS|ANYINTERACT|ON'
    mask_regex = re.compile('^(%s)(\\+(%s))*$' % (masks, masks), re.I)
    if not isinstance(arg, str) or not mask_regex.match(arg):
        raise ValueError('Invalid SDO_RELATE mask: "%s"' % arg)