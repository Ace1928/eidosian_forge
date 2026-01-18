from django.contrib.gis.db.models import GeometryField
from django.contrib.gis.db.models.functions import Distance
from django.contrib.gis.measure import Area as AreaMeasure
from django.contrib.gis.measure import Distance as DistanceMeasure
from django.db import NotSupportedError
from django.utils.functional import cached_property
@cached_property
def select_extent(self):
    return self.select