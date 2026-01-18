from aniso8601.builders import DatetimeTuple, DateTuple, TupleBuilder
from aniso8601.builders.python import PythonTimeBuilder
from aniso8601.compat import is_string
from aniso8601.date import parse_date
from aniso8601.duration import parse_duration
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import IntervalResolution
from aniso8601.time import parse_datetime, parse_time
def parse_interval(isointervalstr, intervaldelimiter='/', datetimedelimiter='T', builder=PythonTimeBuilder):
    if is_string(isointervalstr) is False:
        raise ValueError('Interval must be string.')
    if len(isointervalstr) == 0:
        raise ISOFormatError('Interval string is empty.')
    if isointervalstr[0] == 'R':
        raise ISOFormatError('ISO 8601 repeating intervals must be parsed with parse_repeating_interval.')
    intervaldelimitercount = isointervalstr.count(intervaldelimiter)
    if intervaldelimitercount == 0:
        raise ISOFormatError('Interval delimiter "{0}" is not in interval string "{1}".'.format(intervaldelimiter, isointervalstr))
    if intervaldelimitercount > 1:
        raise ISOFormatError('{0} is not a valid ISO 8601 interval'.format(isointervalstr))
    return _parse_interval(isointervalstr, builder, intervaldelimiter, datetimedelimiter)