import copy
import difflib
import functools
import sys
from collections import Counter, namedtuple
from collections.abc import Iterator, Mapping
from itertools import chain, count, product
from string import ascii_uppercase
from django.core.exceptions import FieldDoesNotExist, FieldError
from django.db import DEFAULT_DB_ALIAS, NotSupportedError, connections
from django.db.models.aggregates import Count
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import (
from django.db.models.fields import Field
from django.db.models.fields.related_lookups import MultiColSource
from django.db.models.lookups import Lookup
from django.db.models.query_utils import (
from django.db.models.sql.constants import INNER, LOUTER, ORDER_DIR, SINGLE
from django.db.models.sql.datastructures import BaseTable, Empty, Join, MultiJoin
from django.db.models.sql.where import AND, OR, ExtraWhere, NothingNode, WhereNode
from django.utils.functional import cached_property
from django.utils.regex_helper import _lazy_re_compile
from django.utils.tree import Node
def check_related_objects(self, field, value, opts):
    """Check the type of object passed to query relations."""
    if field.is_relation:
        if isinstance(value, Query) and (not value.has_select_fields) and (not check_rel_lookup_compatibility(value.model, opts, field)):
            raise ValueError('Cannot use QuerySet for "%s": Use a QuerySet for "%s".' % (value.model._meta.object_name, opts.object_name))
        elif hasattr(value, '_meta'):
            self.check_query_object_type(value, opts, field)
        elif hasattr(value, '__iter__'):
            for v in value:
                self.check_query_object_type(v, opts, field)