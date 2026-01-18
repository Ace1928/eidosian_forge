import uuid
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta
from django.db.models import Exists, ExpressionWrapper, Lookup
from django.db.models.constants import OnConflict
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.regex_helper import _lazy_re_compile
def date_trunc_sql(self, lookup_type, sql, params, tzname=None):
    sql, params = self._convert_sql_to_tz(sql, params, tzname)
    fields = {'year': '%Y-01-01', 'month': '%Y-%m-01'}
    if lookup_type in fields:
        format_str = fields[lookup_type]
        return (f'CAST(DATE_FORMAT({sql}, %s) AS DATE)', (*params, format_str))
    elif lookup_type == 'quarter':
        return (f'MAKEDATE(YEAR({sql}), 1) + INTERVAL QUARTER({sql}) QUARTER - INTERVAL 1 QUARTER', (*params, *params))
    elif lookup_type == 'week':
        return (f'DATE_SUB({sql}, INTERVAL WEEKDAY({sql}) DAY)', (*params, *params))
    else:
        return (f'DATE({sql})', params)