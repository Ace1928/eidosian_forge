from django.apps.registry import Apps
from django.db import DatabaseError, models
from django.utils.functional import classproperty
from django.utils.timezone import now
from .exceptions import MigrationSchemaMissing
def applied_migrations(self):
    """
        Return a dict mapping (app_name, migration_name) to Migration instances
        for all applied migrations.
        """
    if self.has_table():
        return {(migration.app, migration.name): migration for migration in self.migration_qs}
    else:
        return {}