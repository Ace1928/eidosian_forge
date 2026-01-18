from django.contrib.gis.db.models import GeometryField
from django.contrib.gis.db.models.functions import Distance
from django.contrib.gis.measure import Area as AreaMeasure
from django.contrib.gis.measure import Distance as DistanceMeasure
from django.db import NotSupportedError
from django.utils.functional import cached_property
def check_expression_support(self, expression):
    if isinstance(expression, self.disallowed_aggregates):
        raise NotSupportedError('%s spatial aggregation is not supported by this database backend.' % expression.name)
    super().check_expression_support(expression)