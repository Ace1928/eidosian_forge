import json
from functools import lru_cache, partial
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.postgresql.psycopg_any import (
from django.db.backends.utils import split_tzname_delta
from django.db.models.constants import OnConflict
from django.db.models.functions import Cast
from django.utils.regex_helper import _lazy_re_compile
def sequence_reset_sql(self, style, model_list):
    from django.db import models
    output = []
    qn = self.quote_name
    for model in model_list:
        for f in model._meta.local_fields:
            if isinstance(f, models.AutoField):
                output.append("%s setval(pg_get_serial_sequence('%s','%s'), coalesce(max(%s), 1), max(%s) %s null) %s %s;" % (style.SQL_KEYWORD('SELECT'), style.SQL_TABLE(qn(model._meta.db_table)), style.SQL_FIELD(f.column), style.SQL_FIELD(qn(f.column)), style.SQL_FIELD(qn(f.column)), style.SQL_KEYWORD('IS NOT'), style.SQL_KEYWORD('FROM'), style.SQL_TABLE(qn(model._meta.db_table))))
                break
    return output