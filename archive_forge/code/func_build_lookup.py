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
def build_lookup(self, lookups, lhs, rhs):
    """
        Try to extract transforms and lookup from given lhs.

        The lhs value is something that works like SQLExpression.
        The rhs value is what the lookup is going to compare against.
        The lookups is a list of names to extract using get_lookup()
        and get_transform().
        """
    *transforms, lookup_name = lookups or ['exact']
    for name in transforms:
        lhs = self.try_transform(lhs, name)
    lookup_class = lhs.get_lookup(lookup_name)
    if not lookup_class:
        lhs = self.try_transform(lhs, lookup_name)
        lookup_name = 'exact'
        lookup_class = lhs.get_lookup(lookup_name)
        if not lookup_class:
            return
    lookup = lookup_class(lhs, rhs)
    if lookup.rhs is None and (not lookup.can_use_none_as_rhs):
        if lookup_name not in ('exact', 'iexact'):
            raise ValueError('Cannot use None as a query value')
        return lhs.get_lookup('isnull')(lhs, True)
    if lookup_name == 'exact' and lookup.rhs == '' and connections[DEFAULT_DB_ALIAS].features.interprets_empty_strings_as_nulls:
        return lhs.get_lookup('isnull')(lhs, True)
    return lookup