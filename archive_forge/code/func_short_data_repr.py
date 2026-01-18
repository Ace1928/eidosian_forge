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
def short_data_repr(array):
    """Format "data" for DataArray and Variable."""
    internal_data = getattr(array, 'variable', array)._data
    if isinstance(array, np.ndarray):
        return short_array_repr(array)
    elif is_duck_array(internal_data):
        return limit_lines(repr(array.data), limit=40)
    elif getattr(array, '_in_memory', None):
        return short_array_repr(array)
    else:
        return f'[{array.size} values with dtype={array.dtype}]'