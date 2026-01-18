from django.db.migrations.utils import field_references
from django.db.models import NOT_PROVIDED
from django.utils.functional import cached_property
from .base import Operation
def references_field(self, model_name, name, app_label):
    return self.references_model(model_name, app_label) and (name.lower() == self.old_name_lower or name.lower() == self.new_name_lower)