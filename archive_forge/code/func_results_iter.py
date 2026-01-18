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
def results_iter(self, results=None, tuple_expected=False, chunked_fetch=False, chunk_size=GET_ITERATOR_CHUNK_SIZE):
    """Return an iterator over the results from executing this query."""
    if results is None:
        results = self.execute_sql(MULTI, chunked_fetch=chunked_fetch, chunk_size=chunk_size)
    fields = [s[0] for s in self.select[0:self.col_count]]
    converters = self.get_converters(fields)
    rows = chain.from_iterable(results)
    if converters:
        rows = self.apply_converters(rows, converters)
        if tuple_expected:
            rows = map(tuple, rows)
    return rows