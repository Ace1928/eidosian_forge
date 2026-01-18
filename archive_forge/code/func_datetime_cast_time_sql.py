import uuid
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta
from django.db.models import Exists, ExpressionWrapper, Lookup
from django.db.models.constants import OnConflict
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.regex_helper import _lazy_re_compile
def datetime_cast_time_sql(self, sql, params, tzname):
    sql, params = self._convert_sql_to_tz(sql, params, tzname)
    return (f'TIME({sql})', params)