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
def get_from_clause(self):
    """
        Return a list of strings that are joined together to go after the
        "FROM" part of the query, as well as a list any extra parameters that
        need to be included. Subclasses, can override this to create a
        from-clause via a "select".

        This should only be called after any SQL construction methods that
        might change the tables that are needed. This means the select columns,
        ordering, and distinct must be done first.
        """
    result = []
    params = []
    for alias in tuple(self.query.alias_map):
        if not self.query.alias_refcount[alias]:
            continue
        try:
            from_clause = self.query.alias_map[alias]
        except KeyError:
            continue
        clause_sql, clause_params = self.compile(from_clause)
        result.append(clause_sql)
        params.extend(clause_params)
    for t in self.query.extra_tables:
        alias, _ = self.query.table_alias(t)
        if alias not in self.query.alias_map or self.query.alias_refcount[alias] == 1:
            result.append(', %s' % self.quote_name_unless_alias(alias))
    return (result, params)