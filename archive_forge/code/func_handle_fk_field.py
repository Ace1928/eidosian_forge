from django.apps import apps
from django.core.serializers import base
from django.db import DEFAULT_DB_ALIAS, models
from django.utils.encoding import is_protected_type
def handle_fk_field(self, obj, field):
    if self.use_natural_foreign_keys and hasattr(field.remote_field.model, 'natural_key'):
        related = getattr(obj, field.name)
        if related:
            value = related.natural_key()
        else:
            value = None
    else:
        value = self._value_from_field(obj, field)
    self._current[field.name] = value