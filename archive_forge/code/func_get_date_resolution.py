from aniso8601.builders import TupleBuilder
from aniso8601.builders.python import PythonTimeBuilder
from aniso8601.compat import is_string
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DateResolution
def get_date_resolution(isodatestr):
    isodatetuple = parse_date(isodatestr, builder=TupleBuilder)
    if isodatetuple.DDD is not None:
        return DateResolution.Ordinal
    if isodatetuple.D is not None:
        return DateResolution.Weekday
    if isodatetuple.Www is not None:
        return DateResolution.Week
    if isodatetuple.DD is not None:
        return DateResolution.Day
    if isodatetuple.MM is not None:
        return DateResolution.Month
    return DateResolution.Year