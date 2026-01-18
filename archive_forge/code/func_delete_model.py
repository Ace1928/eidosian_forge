from django.db import DatabaseError
from django.db.backends.sqlite3.schema import DatabaseSchemaEditor
def delete_model(self, model, **kwargs):
    from django.contrib.gis.db.models import GeometryField
    for field in model._meta.local_fields:
        if isinstance(field, GeometryField):
            self.remove_geometry_metadata(model, field)
    for geom_table in self.geometry_tables:
        try:
            self.execute(self.sql_discard_geometry_columns % {'geom_table': geom_table, 'table': self.quote_name(model._meta.db_table)})
        except DatabaseError:
            pass
    super().delete_model(model, **kwargs)