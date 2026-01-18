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
def format_attrs(index, separator=', '):
    """Format attributes of CFTimeIndex for __repr__."""
    attrs = {'dtype': f"'{index.dtype}'", 'length': f'{len(index)}', 'calendar': f'{index.calendar!r}', 'freq': f'{index.freq!r}'}
    attrs_str = [f'{k}={v}' for k, v in attrs.items()]
    attrs_str = f'{separator}'.join(attrs_str)
    return attrs_str