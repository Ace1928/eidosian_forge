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
def check_fid_range(self, fid_range):
    """Check the `fid_range` keyword."""
    if fid_range:
        if isinstance(fid_range, (tuple, list)):
            return slice(*fid_range)
        elif isinstance(fid_range, slice):
            return fid_range
        else:
            raise TypeError
    else:
        return None