import os
import sys
from io import StringIO
from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.db import router
from django.db.transaction import atomic
from django.utils.module_loading import import_string
def serialize_db_to_string(self):
    """
        Serialize all data in the database into a JSON string.
        Designed only for test runner usage; will not handle large
        amounts of data.
        """

    def get_objects():
        from django.db.migrations.loader import MigrationLoader
        loader = MigrationLoader(self.connection)
        for app_config in apps.get_app_configs():
            if app_config.models_module is not None and app_config.label in loader.migrated_apps and (app_config.name not in settings.TEST_NON_SERIALIZED_APPS):
                for model in app_config.get_models():
                    if model._meta.can_migrate(self.connection) and router.allow_migrate_model(self.connection.alias, model):
                        queryset = model._base_manager.using(self.connection.alias).order_by(model._meta.pk.name)
                        chunk_size = 2000 if queryset._prefetch_related_lookups else None
                        yield from queryset.iterator(chunk_size=chunk_size)
    out = StringIO()
    serializers.serialize('json', get_objects(), indent=None, stream=out)
    return out.getvalue()