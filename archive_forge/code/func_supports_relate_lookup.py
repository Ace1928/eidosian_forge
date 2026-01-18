import re
from django.contrib.gis.db import models
@property
def supports_relate_lookup(self):
    return 'relate' in self.connection.ops.gis_operators