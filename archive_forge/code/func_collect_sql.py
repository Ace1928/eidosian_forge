import pkgutil
import sys
from importlib import import_module, reload
from django.apps import apps
from django.conf import settings
from django.db.migrations.graph import MigrationGraph
from django.db.migrations.recorder import MigrationRecorder
from .exceptions import (
def collect_sql(self, plan):
    """
        Take a migration plan and return a list of collected SQL statements
        that represent the best-efforts version of that plan.
        """
    statements = []
    state = None
    for migration, backwards in plan:
        with self.connection.schema_editor(collect_sql=True, atomic=migration.atomic) as schema_editor:
            if state is None:
                state = self.project_state((migration.app_label, migration.name), at_end=False)
            if not backwards:
                state = migration.apply(state, schema_editor, collect_sql=True)
            else:
                state = migration.unapply(state, schema_editor, collect_sql=True)
        statements.extend(schema_editor.collected_sql)
    return statements