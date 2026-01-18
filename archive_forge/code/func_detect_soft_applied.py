from django.apps.registry import apps as global_apps
from django.db import migrations, router
from .exceptions import InvalidMigrationPlan
from .loader import MigrationLoader
from .recorder import MigrationRecorder
from .state import ProjectState
def detect_soft_applied(self, project_state, migration):
    """
        Test whether a migration has been implicitly applied - that the
        tables or columns it would create exist. This is intended only for use
        on initial migrations (as it only looks for CreateModel and AddField).
        """

    def should_skip_detecting_model(migration, model):
        """
            No need to detect tables for proxy models, unmanaged models, or
            models that can't be migrated on the current database.
            """
        return model._meta.proxy or not model._meta.managed or (not router.allow_migrate(self.connection.alias, migration.app_label, model_name=model._meta.model_name))
    if migration.initial is None:
        if any((app == migration.app_label for app, name in migration.dependencies)):
            return (False, project_state)
    elif migration.initial is False:
        return (False, project_state)
    if project_state is None:
        after_state = self.loader.project_state((migration.app_label, migration.name), at_end=True)
    else:
        after_state = migration.mutate_state(project_state)
    apps = after_state.apps
    found_create_model_migration = False
    found_add_field_migration = False
    fold_identifier_case = self.connection.features.ignores_table_name_case
    with self.connection.cursor() as cursor:
        existing_table_names = set(self.connection.introspection.table_names(cursor))
        if fold_identifier_case:
            existing_table_names = {name.casefold() for name in existing_table_names}
    for operation in migration.operations:
        if isinstance(operation, migrations.CreateModel):
            model = apps.get_model(migration.app_label, operation.name)
            if model._meta.swapped:
                model = global_apps.get_model(model._meta.swapped)
            if should_skip_detecting_model(migration, model):
                continue
            db_table = model._meta.db_table
            if fold_identifier_case:
                db_table = db_table.casefold()
            if db_table not in existing_table_names:
                return (False, project_state)
            found_create_model_migration = True
        elif isinstance(operation, migrations.AddField):
            model = apps.get_model(migration.app_label, operation.model_name)
            if model._meta.swapped:
                model = global_apps.get_model(model._meta.swapped)
            if should_skip_detecting_model(migration, model):
                continue
            table = model._meta.db_table
            field = model._meta.get_field(operation.name)
            if field.many_to_many:
                through_db_table = field.remote_field.through._meta.db_table
                if fold_identifier_case:
                    through_db_table = through_db_table.casefold()
                if through_db_table not in existing_table_names:
                    return (False, project_state)
                else:
                    found_add_field_migration = True
                    continue
            with self.connection.cursor() as cursor:
                columns = self.connection.introspection.get_table_description(cursor, table)
            for column in columns:
                field_column = field.column
                column_name = column.name
                if fold_identifier_case:
                    column_name = column_name.casefold()
                    field_column = field_column.casefold()
                if column_name == field_column:
                    found_add_field_migration = True
                    break
            else:
                return (False, project_state)
    return (found_create_model_migration or found_add_field_migration, after_state)