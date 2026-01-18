from decimal import Decimal
from django.contrib.gis.db.models.fields import BaseSpatialField, GeometryField
from django.contrib.gis.db.models.sql import AreaField, DistanceField
from django.contrib.gis.geos import GEOSGeometry
from django.core.exceptions import FieldError
from django.db import NotSupportedError
from django.db.models import (
from django.db.models.functions import Cast
from django.utils.functional import cached_property
def as_mysql(self, compiler, connection, **extra_context):
    clone = self.copy()
    if len(clone.source_expressions) < 2:
        clone.source_expressions.append(Value(100))
    return clone.as_sql(compiler, connection, **extra_context)