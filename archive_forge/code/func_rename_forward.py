from django.apps import apps as global_apps
from django.db import DEFAULT_DB_ALIAS, IntegrityError, migrations, router, transaction
def rename_forward(self, apps, schema_editor):
    self._rename(apps, schema_editor, self.old_model, self.new_model)