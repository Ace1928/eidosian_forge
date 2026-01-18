import datetime
from decimal import Decimal
import re
import time
from .err import ProgrammingError
from .constants import FIELD_TYPE
def escape_bytes(value, mapping=None):
    return "'%s'" % value.decode('ascii', 'surrogateescape').translate(_escape_table)