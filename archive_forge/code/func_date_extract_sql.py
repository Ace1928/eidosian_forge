import uuid
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta
from django.db.models import Exists, ExpressionWrapper, Lookup
from django.db.models.constants import OnConflict
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.regex_helper import _lazy_re_compile
def date_extract_sql(self, lookup_type, sql, params):
    if lookup_type == 'week_day':
        return (f'DAYOFWEEK({sql})', params)
    elif lookup_type == 'iso_week_day':
        return (f'WEEKDAY({sql}) + 1', params)
    elif lookup_type == 'week':
        return (f'WEEK({sql}, 3)', params)
    elif lookup_type == 'iso_year':
        return (f'TRUNCATE(YEARWEEK({sql}, 3), -2) / 100', params)
    else:
        lookup_type = lookup_type.upper()
        if not self._extract_format_re.fullmatch(lookup_type):
            raise ValueError(f'Invalid loookup type: {lookup_type!r}')
        return (f'EXTRACT({lookup_type} FROM {sql})', params)