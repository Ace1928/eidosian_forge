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
def collect_replacements(expressions):
    while expressions:
        expr = expressions.pop()
        if expr in replacements:
            continue
        elif (select_alias := select.get(expr)):
            replacements[expr] = select_alias
        elif isinstance(expr, Lookup):
            expressions.extend(expr.get_source_expressions())
        elif isinstance(expr, Ref):
            if expr.refs not in select_aliases:
                expressions.extend(expr.get_source_expressions())
        else:
            num_qual_alias = len(qual_aliases)
            select_alias = f'qual{num_qual_alias}'
            qual_aliases.add(select_alias)
            inner_query.add_annotation(expr, select_alias)
            replacements[expr] = select_alias