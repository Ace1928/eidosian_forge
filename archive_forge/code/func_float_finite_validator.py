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
def float_finite_validator(v: 'Number', field: 'ModelField', config: 'BaseConfig') -> 'Number':
    allow_inf_nan = getattr(field.type_, 'allow_inf_nan', None)
    if allow_inf_nan is None:
        allow_inf_nan = config.allow_inf_nan
    if allow_inf_nan is False and (math.isnan(v) or math.isinf(v)):
        raise errors.NumberNotFiniteError()
    return v