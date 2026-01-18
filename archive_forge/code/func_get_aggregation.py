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
def get_aggregation(self, using, aggregate_exprs):
    """
        Return the dictionary with the values of the existing aggregations.
        """
    if not aggregate_exprs:
        return {}
    refs_subquery = False
    refs_window = False
    replacements = {}
    annotation_select_mask = self.annotation_select_mask
    for alias, aggregate_expr in aggregate_exprs.items():
        self.check_alias(alias)
        aggregate = aggregate_expr.resolve_expression(self, allow_joins=True, reuse=None, summarize=True)
        if not aggregate.contains_aggregate:
            raise TypeError('%s is not an aggregate expression' % alias)
        self.append_annotation_mask([alias])
        refs_subquery |= any((getattr(self.annotations[ref], 'contains_subquery', False) for ref in aggregate.get_refs()))
        refs_window |= any((getattr(self.annotations[ref], 'contains_over_clause', True) for ref in aggregate.get_refs()))
        aggregate = aggregate.replace_expressions(replacements)
        self.annotations[alias] = aggregate
        replacements[Ref(alias, aggregate)] = aggregate
    aggregates = {alias: self.annotations.pop(alias) for alias in aggregate_exprs}
    self.set_annotation_mask(annotation_select_mask)
    _, having, qualify = self.where.split_having_qualify()
    has_existing_aggregation = any((getattr(annotation, 'contains_aggregate', True) for annotation in self.annotations.values())) or having
    if isinstance(self.group_by, tuple) or self.is_sliced or has_existing_aggregation or refs_subquery or refs_window or qualify or self.distinct or self.combinator:
        from django.db.models.sql.subqueries import AggregateQuery
        inner_query = self.clone()
        inner_query.subquery = True
        outer_query = AggregateQuery(self.model, inner_query)
        inner_query.select_for_update = False
        inner_query.select_related = False
        inner_query.set_annotation_mask(self.annotation_select)
        inner_query.clear_ordering(force=False)
        if not inner_query.distinct:
            if inner_query.default_cols and has_existing_aggregation:
                inner_query.group_by = (self.model._meta.pk.get_col(inner_query.get_initial_alias()),)
            inner_query.default_cols = False
            if not qualify:
                annotation_mask = set()
                if isinstance(self.group_by, tuple):
                    for expr in self.group_by:
                        annotation_mask |= expr.get_refs()
                for aggregate in aggregates.values():
                    annotation_mask |= aggregate.get_refs()
                for annotation_alias, annotation in self.annotation_select.items():
                    if annotation.get_group_by_cols():
                        annotation_mask.add(annotation_alias)
                inner_query.set_annotation_mask(annotation_mask)
        col_refs = {}
        for alias, aggregate in aggregates.items():
            replacements = {}
            for col in self._gen_cols([aggregate], resolve_refs=False):
                if not (col_ref := col_refs.get(col)):
                    index = len(col_refs) + 1
                    col_alias = f'__col{index}'
                    col_ref = Ref(col_alias, col)
                    col_refs[col] = col_ref
                    inner_query.annotations[col_alias] = col
                    inner_query.append_annotation_mask([col_alias])
                replacements[col] = col_ref
            outer_query.annotations[alias] = aggregate.replace_expressions(replacements)
        if inner_query.select == () and (not inner_query.default_cols) and (not inner_query.annotation_select_mask):
            inner_query.select = (self.model._meta.pk.get_col(inner_query.get_initial_alias()),)
    else:
        outer_query = self
        self.select = ()
        self.default_cols = False
        self.extra = {}
        if self.annotations:
            replacements = {Ref(alias, annotation): annotation for alias, annotation in self.annotations.items()}
            self.annotations = {alias: aggregate.replace_expressions(replacements) for alias, aggregate in aggregates.items()}
        else:
            self.annotations = aggregates
        self.set_annotation_mask(aggregates)
    empty_set_result = [expression.empty_result_set_value for expression in outer_query.annotation_select.values()]
    elide_empty = not any((result is NotImplemented for result in empty_set_result))
    outer_query.clear_ordering(force=True)
    outer_query.clear_limits()
    outer_query.select_for_update = False
    outer_query.select_related = False
    compiler = outer_query.get_compiler(using, elide_empty=elide_empty)
    result = compiler.execute_sql(SINGLE)
    if result is None:
        result = empty_set_result
    else:
        converters = compiler.get_converters(outer_query.annotation_select.values())
        result = next(compiler.apply_converters((result,), converters))
    return dict(zip(outer_query.annotation_select, result))