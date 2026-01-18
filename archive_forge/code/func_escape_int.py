import datetime
from decimal import Decimal
import re
import time
from .err import ProgrammingError
from .constants import FIELD_TYPE
def escape_int(value, mapping=None):
    return str(value)