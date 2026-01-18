import re
from django.contrib.gis.db import models
@property
def supports_extent_aggr(self):
    return models.Extent not in self.connection.ops.disallowed_aggregates