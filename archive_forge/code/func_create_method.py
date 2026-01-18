import copy
import inspect
from functools import wraps
from importlib import import_module
from django.db import router
from django.db.models.query import QuerySet
def create_method(name, method):

    @wraps(method)
    def manager_method(self, *args, **kwargs):
        return getattr(self.get_queryset(), name)(*args, **kwargs)
    return manager_method