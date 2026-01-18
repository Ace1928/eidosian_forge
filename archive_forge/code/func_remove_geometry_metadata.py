from django.db import DatabaseError
from django.db.backends.sqlite3.schema import DatabaseSchemaEditor
def remove_geometry_metadata(self, model, field):
    self.execute(self.sql_remove_geometry_metadata % {'table': self.quote_name(model._meta.db_table), 'column': self.quote_name(field.column)})
    self.execute(self.sql_drop_spatial_index % {'table': model._meta.db_table, 'column': field.column})