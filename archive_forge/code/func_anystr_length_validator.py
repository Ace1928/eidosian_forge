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
def anystr_length_validator(v: 'StrBytes', config: 'BaseConfig') -> 'StrBytes':
    v_len = len(v)
    min_length = config.min_anystr_length
    if v_len < min_length:
        raise errors.AnyStrMinLengthError(limit_value=min_length)
    max_length = config.max_anystr_length
    if max_length is not None and v_len > max_length:
        raise errors.AnyStrMaxLengthError(limit_value=max_length)
    return v