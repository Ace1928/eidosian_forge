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
def check_ogr_fld(ogr_map_fld):
    try:
        idx = ogr_fields.index(ogr_map_fld)
    except ValueError:
        raise LayerMapError('Given mapping OGR field "%s" not found in OGR Layer.' % ogr_map_fld)
    return idx