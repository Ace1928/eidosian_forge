import functools
import inspect
import logging
from collections import namedtuple
from django.core.exceptions import FieldError
from django.db import DEFAULT_DB_ALIAS, DatabaseError, connections
from django.db.models.constants import LOOKUP_SEP
from django.utils import tree
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
def get_instance_lookups(self):
    class_lookups = self.get_class_lookups()
    if (instance_lookups := getattr(self, 'instance_lookups', None)):
        return {**class_lookups, **instance_lookups}
    return class_lookups