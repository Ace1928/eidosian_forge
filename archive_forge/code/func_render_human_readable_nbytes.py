from __future__ import annotations
import contextlib
import functools
import math
from collections import defaultdict
from collections.abc import Collection, Hashable, Sequence
from datetime import datetime, timedelta
from itertools import chain, zip_longest
from reprlib import recursive_repr
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime
from xarray.core.duck_array_ops import array_equiv, astype
from xarray.core.indexing import MemoryCachedArray
from xarray.core.options import OPTIONS, _get_boolean_with_default
from xarray.core.utils import is_duck_array
from xarray.namedarray.pycompat import array_type, to_duck_array, to_numpy
def render_human_readable_nbytes(nbytes: int, /, *, attempt_constant_width: bool=False) -> str:
    """Renders simple human-readable byte count representation

    This is only a quick representation that should not be relied upon for precise needs.

    To get the exact byte count, please use the ``nbytes`` attribute directly.

    Parameters
    ----------
    nbytes
        Byte count
    attempt_constant_width
        For reasonable nbytes sizes, tries to render a fixed-width representation.

    Returns
    -------
        Human-readable representation of the byte count
    """
    dividend = float(nbytes)
    divisor = 1000.0
    last_unit_available = UNITS[-1]
    for unit in UNITS:
        if dividend < divisor or unit == last_unit_available:
            break
        dividend /= divisor
    dividend_str = f'{dividend:.0f}'
    unit_str = f'{unit}'
    if attempt_constant_width:
        dividend_str = dividend_str.rjust(3)
        unit_str = unit_str.ljust(2)
    string = f'{dividend_str}{unit_str}'
    return string