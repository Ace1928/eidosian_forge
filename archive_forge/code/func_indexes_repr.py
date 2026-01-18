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
def indexes_repr(indexes, max_rows: int | None=None) -> str:
    col_width = _calculate_col_width(chain.from_iterable(indexes))
    return _mapping_repr(indexes, 'Indexes', summarize_index, 'display_expand_indexes', col_width=col_width, max_rows=max_rows)