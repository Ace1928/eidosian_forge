import datetime
from decimal import Decimal
import re
import time
from .err import ProgrammingError
from .constants import FIELD_TYPE
def escape_set(val, charset, mapping=None):
    return ','.join([escape_item(x, charset, mapping) for x in val])