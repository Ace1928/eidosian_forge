from django.apps.registry import Apps
from django.db import DatabaseError, models
from django.utils.functional import classproperty
from django.utils.timezone import now
from .exceptions import MigrationSchemaMissing
def ensure_schema(self):
    """Ensure the table exists and has the correct schema."""
    if self.has_table():
        return
    try:
        with self.connection.schema_editor() as editor:
            editor.create_model(self.Migration)
    except DatabaseError as exc:
        raise MigrationSchemaMissing('Unable to create the django_migrations table (%s)' % exc)