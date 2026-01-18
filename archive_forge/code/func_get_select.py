import collections
import json
import re
from functools import partial
from itertools import chain
from django.core.exceptions import EmptyResultSet, FieldError, FullResultSet
from django.db import DatabaseError, NotSupportedError
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import F, OrderBy, RawSQL, Ref, Value
from django.db.models.functions import Cast, Random
from django.db.models.lookups import Lookup
from django.db.models.query_utils import select_related_descend
from django.db.models.sql.constants import (
from django.db.models.sql.query import Query, get_order_dir
from django.db.models.sql.where import AND
from django.db.transaction import TransactionManagementError
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
from django.utils.regex_helper import _lazy_re_compile
def get_select(self, with_col_aliases=False):
    """
        Return three values:
        - a list of 3-tuples of (expression, (sql, params), alias)
        - a klass_info structure,
        - a dictionary of annotations

        The (sql, params) is what the expression will produce, and alias is the
        "AS alias" for the column (possibly None).

        The klass_info structure contains the following information:
        - The base model of the query.
        - Which columns for that model are present in the query (by
          position of the select clause).
        - related_klass_infos: [f, klass_info] to descent into

        The annotations is a dictionary of {'attname': column position} values.
        """
    select = []
    klass_info = None
    annotations = {}
    select_idx = 0
    for alias, (sql, params) in self.query.extra_select.items():
        annotations[alias] = select_idx
        select.append((RawSQL(sql, params), alias))
        select_idx += 1
    assert not (self.query.select and self.query.default_cols)
    select_mask = self.query.get_select_mask()
    if self.query.default_cols:
        cols = self.get_default_columns(select_mask)
    else:
        cols = self.query.select
    if cols:
        select_list = []
        for col in cols:
            select_list.append(select_idx)
            select.append((col, None))
            select_idx += 1
        klass_info = {'model': self.query.model, 'select_fields': select_list}
    for alias, annotation in self.query.annotation_select.items():
        annotations[alias] = select_idx
        select.append((annotation, alias))
        select_idx += 1
    if self.query.select_related:
        related_klass_infos = self.get_related_selections(select, select_mask)
        klass_info['related_klass_infos'] = related_klass_infos

        def get_select_from_parent(klass_info):
            for ki in klass_info['related_klass_infos']:
                if ki['from_parent']:
                    ki['select_fields'] = klass_info['select_fields'] + ki['select_fields']
                get_select_from_parent(ki)
        get_select_from_parent(klass_info)
    ret = []
    col_idx = 1
    for col, alias in select:
        try:
            sql, params = self.compile(col)
        except EmptyResultSet:
            empty_result_set_value = getattr(col, 'empty_result_set_value', NotImplemented)
            if empty_result_set_value is NotImplemented:
                sql, params = ('0', ())
            else:
                sql, params = self.compile(Value(empty_result_set_value))
        except FullResultSet:
            sql, params = self.compile(Value(True))
        else:
            sql, params = col.select_format(self, sql, params)
        if alias is None and with_col_aliases:
            alias = f'col{col_idx}'
            col_idx += 1
        ret.append((col, (sql, params), alias))
    return (ret, klass_info, annotations)