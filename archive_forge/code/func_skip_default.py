import logging
from django.contrib.gis.db.models import GeometryField
from django.db import OperationalError
from django.db.backends.mysql.schema import DatabaseSchemaEditor
def skip_default(self, field):
    if isinstance(field, GeometryField) and (not self._supports_limited_data_type_defaults):
        return True
    return super().skip_default(field)