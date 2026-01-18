from django.apps import apps
from django.core.serializers import base
from django.db import DEFAULT_DB_ALIAS, models
from django.utils.encoding import is_protected_type
def m2m_value(value):
    return self._value_from_field(value, value._meta.pk)