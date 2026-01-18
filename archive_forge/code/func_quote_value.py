import logging
from django.contrib.gis.db.models import GeometryField
from django.db import OperationalError
from django.db.backends.mysql.schema import DatabaseSchemaEditor
def quote_value(self, value):
    if isinstance(value, self.connection.ops.Adapter):
        return super().quote_value(str(value))
    return super().quote_value(value)