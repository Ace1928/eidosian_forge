from abc import abstractmethod
import math
import operator
import re
import datetime
from calendar import isleap
from decimal import Decimal, Context
from typing import cast, Any, Callable, Dict, Optional, Tuple, Union
from ..helpers import MONTH_DAYS_LEAP, MONTH_DAYS, DAYS_IN_4Y, \
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
@property
def iso_year(self) -> str:
    """The ISO string representation of the year field."""
    year = self.year
    if -9999 <= year < -1:
        return '{:05}'.format(year if self.xsd_version == '1.0' else year + 1)
    elif year == -1:
        return '-0001' if self.xsd_version == '1.0' else '0000'
    elif 0 <= year <= 9999:
        return '{:04}'.format(year)
    else:
        return str(year)