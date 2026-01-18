import pkgutil
import sys
from importlib import import_module, reload
from django.apps import apps
from django.conf import settings
from django.db.migrations.graph import MigrationGraph
from django.db.migrations.recorder import MigrationRecorder
from .exceptions import (
def get_migration_by_prefix(self, app_label, name_prefix):
    """
        Return the migration(s) which match the given app label and name_prefix.
        """
    results = []
    for migration_app_label, migration_name in self.disk_migrations:
        if migration_app_label == app_label and migration_name.startswith(name_prefix):
            results.append((migration_app_label, migration_name))
    if len(results) > 1:
        raise AmbiguityError("There is more than one migration for '%s' with the prefix '%s'" % (app_label, name_prefix))
    elif not results:
        raise KeyError(f"There is no migration for '{app_label}' with the prefix '{name_prefix}'")
    else:
        return self.disk_migrations[results[0]]