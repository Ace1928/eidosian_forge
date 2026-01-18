import datetime
from django.utils.regex_helper import _lazy_re_compile
from django.utils.timezone import get_fixed_timezone
def parse_time(value):
    """Parse a string and return a datetime.time.

    This function doesn't support time zone offsets.

    Raise ValueError if the input is well formatted but not a valid time.
    Return None if the input isn't well formatted, in particular if it
    contains an offset.
    """
    try:
        return datetime.time.fromisoformat(value).replace(tzinfo=None)
    except ValueError:
        if (match := time_re.match(value)):
            kw = match.groupdict()
            kw['microsecond'] = kw['microsecond'] and kw['microsecond'].ljust(6, '0')
            kw = {k: int(v) for k, v in kw.items() if v is not None}
            return datetime.time(**kw)