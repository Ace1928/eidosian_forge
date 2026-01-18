import uuid
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta
from django.db.models import Exists, ExpressionWrapper, Lookup
from django.db.models.constants import OnConflict
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.regex_helper import _lazy_re_compile
def datetime_trunc_sql(self, lookup_type, sql, params, tzname):
    sql, params = self._convert_sql_to_tz(sql, params, tzname)
    fields = ['year', 'month', 'day', 'hour', 'minute', 'second']
    format = ('%Y-', '%m', '-%d', ' %H:', '%i', ':%s')
    format_def = ('0000-', '01', '-01', ' 00:', '00', ':00')
    if lookup_type == 'quarter':
        return (f'CAST(DATE_FORMAT(MAKEDATE(YEAR({sql}), 1) + INTERVAL QUARTER({sql}) QUARTER - INTERVAL 1 QUARTER, %s) AS DATETIME)', (*params, *params, '%Y-%m-01 00:00:00'))
    if lookup_type == 'week':
        return (f'CAST(DATE_FORMAT(DATE_SUB({sql}, INTERVAL WEEKDAY({sql}) DAY), %s) AS DATETIME)', (*params, *params, '%Y-%m-%d 00:00:00'))
    try:
        i = fields.index(lookup_type) + 1
    except ValueError:
        pass
    else:
        format_str = ''.join(format[:i] + format_def[i:])
        return (f'CAST(DATE_FORMAT({sql}, %s) AS DATETIME)', (*params, format_str))
    return (sql, params)