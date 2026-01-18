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
def collapse_group_by(self, expressions, having):
    if self.connection.features.allows_group_by_selected_pks:
        pks = {expr for expr in expressions if hasattr(expr, 'target') and expr.target.primary_key and self.connection.features.allows_group_by_selected_pks_on_model(expr.target.model)}
        aliases = {expr.alias for expr in pks}
        expressions = [expr for expr in expressions if expr in pks or expr in having or getattr(expr, 'alias', None) not in aliases]
    return expressions