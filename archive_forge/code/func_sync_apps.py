import sys
import time
from importlib import import_module
from django.apps import apps
from django.core.management.base import BaseCommand, CommandError, no_translations
from django.core.management.sql import emit_post_migrate_signal, emit_pre_migrate_signal
from django.db import DEFAULT_DB_ALIAS, connections, router
from django.db.migrations.autodetector import MigrationAutodetector
from django.db.migrations.executor import MigrationExecutor
from django.db.migrations.loader import AmbiguityError
from django.db.migrations.state import ModelState, ProjectState
from django.utils.module_loading import module_has_submodule
from django.utils.text import Truncator
def sync_apps(self, connection, app_labels):
    """Run the old syncdb-style operation on a list of app_labels."""
    with connection.cursor() as cursor:
        tables = connection.introspection.table_names(cursor)
    all_models = [(app_config.label, router.get_migratable_models(app_config, connection.alias, include_auto_created=False)) for app_config in apps.get_app_configs() if app_config.models_module is not None and app_config.label in app_labels]

    def model_installed(model):
        opts = model._meta
        converter = connection.introspection.identifier_converter
        return not (converter(opts.db_table) in tables or (opts.auto_created and converter(opts.auto_created._meta.db_table) in tables))
    manifest = {app_name: list(filter(model_installed, model_list)) for app_name, model_list in all_models}
    if self.verbosity >= 1:
        self.stdout.write('  Creating tables...')
    with connection.schema_editor() as editor:
        for app_name, model_list in manifest.items():
            for model in model_list:
                if not model._meta.can_migrate(connection):
                    continue
                if self.verbosity >= 3:
                    self.stdout.write('    Processing %s.%s model' % (app_name, model._meta.object_name))
                if self.verbosity >= 1:
                    self.stdout.write('    Creating table %s' % model._meta.db_table)
                editor.create_model(model)
        if self.verbosity >= 1:
            self.stdout.write('    Running deferred SQL...')