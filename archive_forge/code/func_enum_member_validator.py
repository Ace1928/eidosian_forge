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
def enum_member_validator(v: Any, field: 'ModelField', config: 'BaseConfig') -> Enum:
    try:
        enum_v = field.type_(v)
    except ValueError:
        raise errors.EnumMemberError(enum_values=list(field.type_))
    return enum_v.value if config.use_enum_values else enum_v