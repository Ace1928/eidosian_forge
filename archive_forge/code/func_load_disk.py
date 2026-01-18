import pkgutil
import sys
from importlib import import_module, reload
from django.apps import apps
from django.conf import settings
from django.db.migrations.graph import MigrationGraph
from django.db.migrations.recorder import MigrationRecorder
from .exceptions import (
def load_disk(self):
    """Load the migrations from all INSTALLED_APPS from disk."""
    self.disk_migrations = {}
    self.unmigrated_apps = set()
    self.migrated_apps = set()
    for app_config in apps.get_app_configs():
        module_name, explicit = self.migrations_module(app_config.label)
        if module_name is None:
            self.unmigrated_apps.add(app_config.label)
            continue
        was_loaded = module_name in sys.modules
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as e:
            if explicit and self.ignore_no_migrations or (not explicit and MIGRATIONS_MODULE_NAME in e.name.split('.')):
                self.unmigrated_apps.add(app_config.label)
                continue
            raise
        else:
            if not hasattr(module, '__path__'):
                self.unmigrated_apps.add(app_config.label)
                continue
            if getattr(module, '__file__', None) is None and (not isinstance(module.__path__, list)):
                self.unmigrated_apps.add(app_config.label)
                continue
            if was_loaded:
                reload(module)
        self.migrated_apps.add(app_config.label)
        migration_names = {name for _, name, is_pkg in pkgutil.iter_modules(module.__path__) if not is_pkg and name[0] not in '_~'}
        for migration_name in migration_names:
            migration_path = '%s.%s' % (module_name, migration_name)
            try:
                migration_module = import_module(migration_path)
            except ImportError as e:
                if 'bad magic number' in str(e):
                    raise ImportError("Couldn't import %r as it appears to be a stale .pyc file." % migration_path) from e
                else:
                    raise
            if not hasattr(migration_module, 'Migration'):
                raise BadMigrationError('Migration %s in app %s has no Migration class' % (migration_name, app_config.label))
            self.disk_migrations[app_config.label, migration_name] = migration_module.Migration(migration_name, app_config.label)