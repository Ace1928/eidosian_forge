from django.contrib.gis.db.models import GeometryField
from django.db.backends.oracle.schema import DatabaseSchemaEditor
from django.db.backends.utils import strip_quotes, truncate_name
def run_geometry_sql(self):
    for sql in self.geometry_sql:
        self.execute(sql)
    self.geometry_sql = []