import re
from django.contrib.gis.db import models
@property
def supports_contained_lookup(self):
    return 'contained' in self.connection.ops.gis_operators