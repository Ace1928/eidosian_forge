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
def get_distinct(self):
    """
        Return a quoted list of fields to use in DISTINCT ON part of the query.

        This method can alter the tables in the query, and thus it must be
        called before get_from_clause().
        """
    result = []
    params = []
    opts = self.query.get_meta()
    for name in self.query.distinct_fields:
        parts = name.split(LOOKUP_SEP)
        _, targets, alias, joins, path, _, transform_function = self._setup_joins(parts, opts, None)
        targets, alias, _ = self.query.trim_joins(targets, joins, path)
        for target in targets:
            if name in self.query.annotation_select:
                result.append(self.connection.ops.quote_name(name))
            else:
                r, p = self.compile(transform_function(target, alias))
                result.append(r)
                params.append(p)
    return (result, params)