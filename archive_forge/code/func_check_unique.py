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
def check_unique(self, unique):
    """Check the `unique` keyword parameter -- may be a sequence or string."""
    if isinstance(unique, (list, tuple)):
        for attr in unique:
            if attr not in self.mapping:
                raise ValueError
    elif isinstance(unique, str):
        if unique not in self.mapping:
            raise ValueError
    else:
        raise TypeError('Unique keyword argument must be set with a tuple, list, or string.')