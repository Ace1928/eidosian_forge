import re
from django.contrib.gis.db import models
@property
def supports_isvalid_lookup(self):
    return self.has_IsValid_function