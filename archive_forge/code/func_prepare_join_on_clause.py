import json
from functools import lru_cache, partial
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.postgresql.psycopg_any import (
from django.db.backends.utils import split_tzname_delta
from django.db.models.constants import OnConflict
from django.db.models.functions import Cast
from django.utils.regex_helper import _lazy_re_compile
def prepare_join_on_clause(self, lhs_table, lhs_field, rhs_table, rhs_field):
    lhs_expr, rhs_expr = super().prepare_join_on_clause(lhs_table, lhs_field, rhs_table, rhs_field)
    if lhs_field.db_type(self.connection) != rhs_field.db_type(self.connection):
        rhs_expr = Cast(rhs_expr, lhs_field)
    return (lhs_expr, rhs_expr)