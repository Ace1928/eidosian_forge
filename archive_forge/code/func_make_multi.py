import sys
from decimal import Decimal
from decimal import InvalidOperation as DecimalInvalidOperation
from pathlib import Path
from django.contrib.gis.db.models import GeometryField
from django.contrib.gis.gdal import (
from django.contrib.gis.gdal.field import (
from django.core.exceptions import FieldDoesNotExist, ObjectDoesNotExist
from django.db import connections, models, router, transaction
from django.utils.encoding import force_str
def make_multi(self, geom_type, model_field):
    """
        Given the OGRGeomType for a geometry and its associated GeometryField,
        determine whether the geometry should be turned into a GeometryCollection.
        """
    return geom_type.num in self.MULTI_TYPES and model_field.__class__.__name__ == 'Multi%s' % geom_type.django