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
def get_combinator_sql(self, combinator, all):
    features = self.connection.features
    compilers = [query.get_compiler(self.using, self.connection, self.elide_empty) for query in self.query.combined_queries]
    if not features.supports_slicing_ordering_in_compound:
        for compiler in compilers:
            if compiler.query.is_sliced:
                raise DatabaseError('LIMIT/OFFSET not allowed in subqueries of compound statements.')
            if compiler.get_order_by():
                raise DatabaseError('ORDER BY not allowed in subqueries of compound statements.')
    elif self.query.is_sliced and combinator == 'union':
        for compiler in compilers:
            compiler.elide_empty = False
    parts = ()
    for compiler in compilers:
        try:
            if not compiler.query.values_select and self.query.values_select:
                compiler.query = compiler.query.clone()
                compiler.query.set_values((*self.query.extra_select, *self.query.values_select, *self.query.annotation_select))
            part_sql, part_args = compiler.as_sql(with_col_aliases=True)
            if compiler.query.combinator:
                if not features.supports_parentheses_in_compound:
                    part_sql = 'SELECT * FROM ({})'.format(part_sql)
                elif self.query.subquery or not features.supports_slicing_ordering_in_compound:
                    part_sql = '({})'.format(part_sql)
            elif self.query.subquery and features.supports_slicing_ordering_in_compound:
                part_sql = '({})'.format(part_sql)
            parts += ((part_sql, part_args),)
        except EmptyResultSet:
            if combinator == 'union' or (combinator == 'difference' and parts):
                continue
            raise
    if not parts:
        raise EmptyResultSet
    combinator_sql = self.connection.ops.set_operators[combinator]
    if all and combinator == 'union':
        combinator_sql += ' ALL'
    braces = '{}'
    if not self.query.subquery and features.supports_slicing_ordering_in_compound:
        braces = '({})'
    sql_parts, args_parts = zip(*((braces.format(sql), args) for sql, args in parts))
    result = [' {} '.format(combinator_sql).join(sql_parts)]
    params = []
    for part in args_parts:
        params.extend(part)
    return (result, params)