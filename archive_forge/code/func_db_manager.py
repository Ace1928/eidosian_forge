import copy
import inspect
from functools import wraps
from importlib import import_module
from django.db import router
from django.db.models.query import QuerySet
def db_manager(self, using=None, hints=None):
    obj = copy.copy(self)
    obj._db = using or self._db
    obj._hints = hints or self._hints
    return obj