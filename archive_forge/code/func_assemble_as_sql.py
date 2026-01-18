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
def assemble_as_sql(self, fields, value_rows):
    """
        Take a sequence of N fields and a sequence of M rows of values, and
        generate placeholder SQL and parameters for each field and value.
        Return a pair containing:
         * a sequence of M rows of N SQL placeholder strings, and
         * a sequence of M rows of corresponding parameter values.

        Each placeholder string may contain any number of '%s' interpolation
        strings, and each parameter row will contain exactly as many params
        as the total number of '%s's in the corresponding placeholder row.
        """
    if not value_rows:
        return ([], [])
    rows_of_fields_as_sql = ((self.field_as_sql(field, v) for field, v in zip(fields, row)) for row in value_rows)
    sql_and_param_pair_rows = (zip(*row) for row in rows_of_fields_as_sql)
    placeholder_rows, param_rows = zip(*sql_and_param_pair_rows)
    param_rows = [[p for ps in row for p in ps] for row in param_rows]
    return (placeholder_rows, param_rows)