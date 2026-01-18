from aniso8601.builders import DatetimeTuple, DateTuple, TupleBuilder
from aniso8601.builders.python import PythonTimeBuilder
from aniso8601.compat import is_string
from aniso8601.date import parse_date
from aniso8601.duration import parse_duration
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import IntervalResolution
from aniso8601.time import parse_datetime, parse_time
def get_interval_resolution(isointervalstr, intervaldelimiter='/', datetimedelimiter='T'):
    isointervaltuple = parse_interval(isointervalstr, intervaldelimiter=intervaldelimiter, datetimedelimiter=datetimedelimiter, builder=TupleBuilder)
    return _get_interval_resolution(isointervaltuple)