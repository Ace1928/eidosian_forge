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
def find_ordering_name(self, name, opts, alias=None, default_order='ASC', already_seen=None):
    """
        Return the table alias (the name might be ambiguous, the alias will
        not be) and column name for ordering by the given 'name' parameter.
        The 'name' is of the form 'field1__field2__...__fieldN'.
        """
    name, order = get_order_dir(name, default_order)
    descending = order == 'DESC'
    pieces = name.split(LOOKUP_SEP)
    field, targets, alias, joins, path, opts, transform_function = self._setup_joins(pieces, opts, alias)
    if field.is_relation and opts.ordering and (getattr(field, 'attname', None) != pieces[-1]) and (name != 'pk') and (not getattr(transform_function, 'has_transforms', False)):
        already_seen = already_seen or set()
        join_tuple = tuple((getattr(self.query.alias_map[j], 'join_cols', None) for j in joins))
        if join_tuple in already_seen:
            raise FieldError('Infinite loop caused by ordering.')
        already_seen.add(join_tuple)
        results = []
        for item in opts.ordering:
            if hasattr(item, 'resolve_expression') and (not isinstance(item, OrderBy)):
                item = item.desc() if descending else item.asc()
            if isinstance(item, OrderBy):
                results.append((item.prefix_references(f'{name}{LOOKUP_SEP}'), False))
                continue
            results.extend(((expr.prefix_references(f'{name}{LOOKUP_SEP}'), is_ref) for expr, is_ref in self.find_ordering_name(item, opts, alias, order, already_seen)))
        return results
    targets, alias, _ = self.query.trim_joins(targets, joins, path)
    return [(OrderBy(transform_function(t, alias), descending=descending), False) for t in targets]