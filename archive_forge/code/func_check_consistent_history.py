import pkgutil
import sys
from importlib import import_module, reload
from django.apps import apps
from django.conf import settings
from django.db.migrations.graph import MigrationGraph
from django.db.migrations.recorder import MigrationRecorder
from .exceptions import (
def check_consistent_history(self, connection):
    """
        Raise InconsistentMigrationHistory if any applied migrations have
        unapplied dependencies.
        """
    recorder = MigrationRecorder(connection)
    applied = recorder.applied_migrations()
    for migration in applied:
        if migration not in self.graph.nodes:
            continue
        for parent in self.graph.node_map[migration].parents:
            if parent not in applied:
                if parent in self.replacements:
                    if all((m in applied for m in self.replacements[parent].replaces)):
                        continue
                raise InconsistentMigrationHistory("Migration {}.{} is applied before its dependency {}.{} on database '{}'.".format(migration[0], migration[1], parent[0], parent[1], connection.alias))