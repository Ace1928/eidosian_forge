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
def split_exclude(self, filter_expr, can_reuse, names_with_path):
    """
        When doing an exclude against any kind of N-to-many relation, we need
        to use a subquery. This method constructs the nested query, given the
        original exclude filter (filter_expr) and the portion up to the first
        N-to-many relation field.

        For example, if the origin filter is ~Q(child__name='foo'), filter_expr
        is ('child__name', 'foo') and can_reuse is a set of joins usable for
        filters in the original query.

        We will turn this into equivalent of:
            WHERE NOT EXISTS(
                SELECT 1
                FROM child
                WHERE name = 'foo' AND child.parent_id = parent.id
                LIMIT 1
            )
        """
    query = self.__class__(self.model)
    query._filtered_relations = self._filtered_relations
    filter_lhs, filter_rhs = filter_expr
    if isinstance(filter_rhs, OuterRef):
        filter_rhs = OuterRef(filter_rhs)
    elif isinstance(filter_rhs, F):
        filter_rhs = OuterRef(filter_rhs.name)
    query.add_filter(filter_lhs, filter_rhs)
    query.clear_ordering(force=True)
    trimmed_prefix, contains_louter = query.trim_start(names_with_path)
    col = query.select[0]
    select_field = col.target
    alias = col.alias
    if alias in can_reuse:
        pk = select_field.model._meta.pk
        query.bump_prefix(self)
        lookup_class = select_field.get_lookup('exact')
        lookup = lookup_class(pk.get_col(query.select[0].alias), pk.get_col(alias))
        query.where.add(lookup, AND)
        query.external_aliases[alias] = True
    else:
        lookup_class = select_field.get_lookup('exact')
        lookup = lookup_class(col, ResolvedOuterRef(trimmed_prefix))
        query.where.add(lookup, AND)
    condition, needed_inner = self.build_filter(Exists(query))
    if contains_louter:
        or_null_condition, _ = self.build_filter(('%s__isnull' % trimmed_prefix, True), current_negated=True, branch_negated=True, can_reuse=can_reuse)
        condition.add(or_null_condition, OR)
    return (condition, needed_inner)