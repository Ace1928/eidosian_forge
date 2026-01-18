from django.contrib.gis.gdal import DataSource
from django.contrib.gis.gdal.field import (
def process_kwarg(kwarg):
    if isinstance(kwarg, (list, tuple)):
        return [s.lower() for s in kwarg]
    elif kwarg:
        return [s.lower() for s in ogr_fields]
    else:
        return []