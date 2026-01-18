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
def field_as_sql(self, field, val):
    """
        Take a field and a value intended to be saved on that field, and
        return placeholder SQL and accompanying params. Check for raw values,
        expressions, and fields with get_placeholder() defined in that order.

        When field is None, consider the value raw and use it as the
        placeholder, with no corresponding parameters returned.
        """
    if field is None:
        sql, params = (val, [])
    elif hasattr(val, 'as_sql'):
        sql, params = self.compile(val)
    elif hasattr(field, 'get_placeholder'):
        sql, params = (field.get_placeholder(val, self, self.connection), [val])
    else:
        sql, params = ('%s', [val])
    params = self.connection.ops.modify_insert_params(sql, params)
    return (sql, params)