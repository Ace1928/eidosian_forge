from django.contrib.gis.db.models.fields import BaseSpatialField
from django.contrib.gis.measure import Distance
from django.db import NotSupportedError
from django.db.models import Expression, Lookup, Transform
from django.db.models.sql.query import Query
from django.utils.regex_helper import _lazy_re_compile
def process_band_indices(self, only_lhs=False):
    """
        Extract the lhs band index from the band transform class and the rhs
        band index from the input tuple.
        """
    if only_lhs:
        self.band_rhs = 1
        self.band_lhs = self.lhs.band_index + 1
        return
    if isinstance(self.lhs, RasterBandTransform):
        self.band_lhs = self.lhs.band_index + 1
    else:
        self.band_lhs = 1
    self.band_rhs, *self.rhs_params = self.rhs_params