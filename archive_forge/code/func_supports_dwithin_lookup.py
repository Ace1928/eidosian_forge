import re
from django.contrib.gis.db import models
@property
def supports_dwithin_lookup(self):
    return 'dwithin' in self.connection.ops.gis_operators