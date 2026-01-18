import re
from django.contrib.gis.db import models
@property
def supports_distances_lookups(self):
    return self.has_Distance_function