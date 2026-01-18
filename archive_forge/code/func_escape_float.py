import datetime
from decimal import Decimal
import re
import time
from .err import ProgrammingError
from .constants import FIELD_TYPE
def escape_float(value, mapping=None):
    s = repr(value)
    if s in ('inf', '-inf', 'nan'):
        raise ProgrammingError('%s can not be used with MySQL' % s)
    if 'e' not in s:
        s += 'e0'
    return s