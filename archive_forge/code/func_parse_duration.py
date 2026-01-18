import datetime
from django.utils.regex_helper import _lazy_re_compile
from django.utils.timezone import get_fixed_timezone
def parse_duration(value):
    """Parse a duration string and return a datetime.timedelta.

    The preferred format for durations in Django is '%d %H:%M:%S.%f'.

    Also supports ISO 8601 representation and PostgreSQL's day-time interval
    format.
    """
    match = standard_duration_re.match(value) or iso8601_duration_re.match(value) or postgres_interval_re.match(value)
    if match:
        kw = match.groupdict()
        sign = -1 if kw.pop('sign', '+') == '-' else 1
        if kw.get('microseconds'):
            kw['microseconds'] = kw['microseconds'].ljust(6, '0')
        kw = {k: float(v.replace(',', '.')) for k, v in kw.items() if v is not None}
        days = datetime.timedelta(kw.pop('days', 0.0) or 0.0)
        if match.re == iso8601_duration_re:
            days *= sign
        return days + sign * datetime.timedelta(**kw)