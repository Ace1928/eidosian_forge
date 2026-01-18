from django.apps import apps as global_apps
from django.db import DEFAULT_DB_ALIAS, IntegrityError, migrations, router, transaction
def rename_backward(self, apps, schema_editor):
    self._rename(apps, schema_editor, self.new_model, self.old_model)