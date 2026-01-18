import datetime
from decimal import Decimal
import re
import time
from .err import ProgrammingError
from .constants import FIELD_TYPE
def escape_struct_time(obj, mapping=None):
    return escape_datetime(datetime.datetime(*obj[:6]))