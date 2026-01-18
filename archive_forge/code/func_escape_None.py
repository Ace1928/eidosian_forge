import datetime
from decimal import Decimal
import re
import time
from .err import ProgrammingError
from .constants import FIELD_TYPE
def escape_None(value, mapping=None):
    return 'NULL'