from django.apps import apps
from django.core.serializers import base
from django.db import DEFAULT_DB_ALIAS, models
from django.utils.encoding import is_protected_type
def handle_m2m_field(self, obj, field):
    if field.remote_field.through._meta.auto_created:
        if self.use_natural_foreign_keys and hasattr(field.remote_field.model, 'natural_key'):

            def m2m_value(value):
                return value.natural_key()

            def queryset_iterator(obj, field):
                return getattr(obj, field.name).iterator()
        else:

            def m2m_value(value):
                return self._value_from_field(value, value._meta.pk)

            def queryset_iterator(obj, field):
                return getattr(obj, field.name).select_related(None).only('pk').iterator()
        m2m_iter = getattr(obj, '_prefetched_objects_cache', {}).get(field.name, queryset_iterator(obj, field))
        self._current[field.name] = [m2m_value(related) for related in m2m_iter]