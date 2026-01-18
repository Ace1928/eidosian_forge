from django.apps import apps
from django.core.serializers import base
from django.db import DEFAULT_DB_ALIAS, models
from django.utils.encoding import is_protected_type
def queryset_iterator(obj, field):
    return getattr(obj, field.name).select_related(None).only('pk').iterator()