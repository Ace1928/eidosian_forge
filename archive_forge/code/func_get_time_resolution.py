from aniso8601.builders import TupleBuilder
from aniso8601.builders.python import PythonTimeBuilder
from aniso8601.compat import is_string
from aniso8601.date import parse_date
from aniso8601.decimalfraction import normalize
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import TimeResolution
from aniso8601.timezone import parse_timezone
def get_time_resolution(isotimestr):
    isotimetuple = parse_time(isotimestr, builder=TupleBuilder)
    return _get_time_resolution(isotimetuple)