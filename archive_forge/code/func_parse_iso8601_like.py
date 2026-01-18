from __future__ import annotations
import math
import re
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.times import (
from xarray.core.common import _contains_cftime_datetimes
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar
def parse_iso8601_like(datetime_string):
    for pattern in _PATTERNS:
        match = re.match(pattern, datetime_string)
        if match:
            return match.groupdict()
    raise ValueError(f'no ISO-8601 or cftime-string-like match for string: {datetime_string}')