import logging
from django.contrib.gis.db.models import GeometryField
from django.db import OperationalError
from django.db.backends.mysql.schema import DatabaseSchemaEditor
def create_spatial_indexes(self):
    for sql in self.geometry_sql:
        try:
            self.execute(sql)
        except OperationalError:
            logger.error(f'Cannot create SPATIAL INDEX {sql}. Only MyISAM, Aria, and InnoDB support them.')
    self.geometry_sql = []