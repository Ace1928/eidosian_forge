from django.apps.registry import apps as global_apps
from django.db import migrations, router
from .exceptions import InvalidMigrationPlan
from .loader import MigrationLoader
from .recorder import MigrationRecorder
from .state import ProjectState
def apply_migration(self, state, migration, fake=False, fake_initial=False):
    """Run a migration forwards."""
    migration_recorded = False
    if self.progress_callback:
        self.progress_callback('apply_start', migration, fake)
    if not fake:
        if fake_initial:
            applied, state = self.detect_soft_applied(state, migration)
            if applied:
                fake = True
        if not fake:
            with self.connection.schema_editor(atomic=migration.atomic) as schema_editor:
                state = migration.apply(state, schema_editor)
                if not schema_editor.deferred_sql:
                    self.record_migration(migration)
                    migration_recorded = True
    if not migration_recorded:
        self.record_migration(migration)
    if self.progress_callback:
        self.progress_callback('apply_success', migration, fake)
    return state