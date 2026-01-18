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
def make_literal_validator(type_: Any) -> Callable[[Any], Any]:
    permitted_choices = all_literal_values(type_)
    allowed_choices = {v: v for v in permitted_choices}

    def literal_validator(v: Any) -> Any:
        try:
            return allowed_choices[v]
        except (KeyError, TypeError):
            raise errors.WrongConstantError(given=v, permitted=permitted_choices)
    return literal_validator