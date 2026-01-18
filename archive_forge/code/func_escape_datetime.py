import datetime
from decimal import Decimal
import re
import time
from .err import ProgrammingError
from .constants import FIELD_TYPE
def escape_datetime(obj, mapping=None):
    if obj.microsecond:
        fmt = "'{0.year:04}-{0.month:02}-{0.day:02}" + " {0.hour:02}:{0.minute:02}:{0.second:02}.{0.microsecond:06}'"
    else:
        fmt = "'{0.year:04}-{0.month:02}-{0.day:02} {0.hour:02}:{0.minute:02}:{0.second:02}'"
    return fmt.format(obj)