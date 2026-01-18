import copy
import inspect
from functools import wraps
from importlib import import_module
from django.db import router
from django.db.models.query import QuerySet
@classmethod
def from_queryset(cls, queryset_class, class_name=None):
    if class_name is None:
        class_name = '%sFrom%s' % (cls.__name__, queryset_class.__name__)
    return type(class_name, (cls,), {'_queryset_class': queryset_class, **cls._get_queryset_methods(queryset_class)})