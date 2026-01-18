from django.contrib.gis.db.models.fields import BaseSpatialField
from django.contrib.gis.measure import Distance
from django.db import NotSupportedError
from django.db.models import Expression, Lookup, Transform
from django.db.models.sql.query import Query
from django.utils.regex_helper import _lazy_re_compile
def process_rhs_params(self):
    if not 1 <= len(self.rhs_params) <= 3:
        raise ValueError("2, 3, or 4-element tuple required for '%s' lookup." % self.lookup_name)
    elif len(self.rhs_params) == 3 and self.rhs_params[2] != 'spheroid':
        raise ValueError("For 4-element tuples the last argument must be the 'spheroid' directive.")
    if len(self.rhs_params) > 1 and self.rhs_params[1] != 'spheroid':
        self.process_band_indices()