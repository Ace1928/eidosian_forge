from aniso8601 import compat
from aniso8601.builders import TupleBuilder
from aniso8601.builders.python import PythonTimeBuilder
from aniso8601.date import parse_date
from aniso8601.decimalfraction import normalize
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.time import parse_time
def get_duration_resolution(isodurationstr):
    isodurationtuple = parse_duration(isodurationstr, builder=TupleBuilder)
    if isodurationtuple.TnS is not None:
        return DurationResolution.Seconds
    if isodurationtuple.TnM is not None:
        return DurationResolution.Minutes
    if isodurationtuple.TnH is not None:
        return DurationResolution.Hours
    if isodurationtuple.PnD is not None:
        return DurationResolution.Days
    if isodurationtuple.PnW is not None:
        return DurationResolution.Weeks
    if isodurationtuple.PnM is not None:
        return DurationResolution.Months
    return DurationResolution.Years