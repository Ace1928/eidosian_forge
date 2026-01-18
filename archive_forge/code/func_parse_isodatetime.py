import builtins
import datetime
import decimal
from http import client as http_client
import pytz
import re
def parse_isodatetime(value):
    if dateutil:
        return dateutil.parser.parse(value)
    m = datetime_re.match(value)
    if m is None:
        raise ValueError("'%s' is not a legal datetime value" % value)
    try:
        ms = 0
        if m.group('sec_frac') is not None:
            f = decimal.Decimal('0.' + m.group('sec_frac'))
            f = f.quantize(decimal.Decimal('0.000001'))
            ms = int(str(f)[2:])
        tz = _parse_tzparts(m.groupdict())
        return datetime.datetime(int(m.group('year')), int(m.group('month')), int(m.group('day')), int(m.group('hour')), int(m.group('min')), int(m.group('sec')), ms, tz)
    except ValueError:
        raise ValueError("'%s' is a out-of-range datetime" % value)