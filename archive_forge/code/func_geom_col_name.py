from django.contrib.gis.db.backends.base.models import SpatialRefSysMixin
from django.db import models
@classmethod
def geom_col_name(cls):
    """
        Return the name of the metadata column used to store the feature
        geometry column.
        """
    return 'f_geometry_column'