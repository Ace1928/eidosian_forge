import math
import re
from collections import OrderedDict, deque
from collections.abc import Hashable as CollectionsHashable
from datetime import date, datetime, time, timedelta
from decimal import Decimal, DecimalException
from enum import Enum, IntEnum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from typing import (
from uuid import UUID
from . import errors
from .datetime_parse import parse_date, parse_datetime, parse_duration, parse_time
from .typing import (
from .utils import almost_equal_floats, lenient_issubclass, sequence_like
def number_multiple_validator(v: 'Number', field: 'ModelField') -> 'Number':
    field_type: ConstrainedNumber = field.type_
    if field_type.multiple_of is not None:
        mod = float(v) / float(field_type.multiple_of) % 1
        if not almost_equal_floats(mod, 0.0) and (not almost_equal_floats(mod, 1.0)):
            raise errors.NumberNotMultipleError(multiple_of=field_type.multiple_of)
    return v