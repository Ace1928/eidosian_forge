from django.contrib.gis.db.models import GeometryField
from django.contrib.gis.db.models.functions import Distance
from django.contrib.gis.measure import Area as AreaMeasure
from django.contrib.gis.measure import Distance as DistanceMeasure
from django.db import NotSupportedError
from django.utils.functional import cached_property
def spatial_function_name(self, func_name):
    if func_name in self.unsupported_functions:
        raise NotSupportedError("This backend doesn't support the %s function." % func_name)
    return self.function_names.get(func_name, self.geom_func_prefix + func_name)