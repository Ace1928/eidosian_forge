from __future__ import annotations
import enum
import functools
import operator
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import timedelta
from html import escape
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
from xarray.core import duck_array_ops
from xarray.core.nputils import NumpyVIndexAdapter
from xarray.core.options import OPTIONS
from xarray.core.types import T_Xarray
from xarray.core.utils import (
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import array_type, integer_types, is_chunked_array
def is_fancy_indexer(indexer: Any) -> bool:
    """Return False if indexer is a int, slice, a 1-dimensional list, or a 0 or
    1-dimensional ndarray; in all other cases return True
    """
    if isinstance(indexer, (int, slice)):
        return False
    if isinstance(indexer, np.ndarray):
        return indexer.ndim > 1
    if isinstance(indexer, list):
        return bool(indexer) and (not isinstance(indexer[0], int))
    return True