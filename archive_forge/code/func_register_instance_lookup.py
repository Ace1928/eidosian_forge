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
def register_instance_lookup(self, lookup, lookup_name=None):
    if lookup_name is None:
        lookup_name = lookup.lookup_name
    if 'instance_lookups' not in self.__dict__:
        self.instance_lookups = {}
    self.instance_lookups[lookup_name] = lookup
    return lookup