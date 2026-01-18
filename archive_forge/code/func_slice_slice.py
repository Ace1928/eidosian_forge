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
def slice_slice(old_slice: slice, applied_slice: slice, size: int) -> slice:
    """Given a slice and the size of the dimension to which it will be applied,
    index it with another slice to return a new slice equivalent to applying
    the slices sequentially
    """
    old_slice = _normalize_slice(old_slice, size)
    size_after_old_slice = len(range(old_slice.start, old_slice.stop, old_slice.step))
    if size_after_old_slice == 0:
        return slice(0)
    applied_slice = _normalize_slice(applied_slice, size_after_old_slice)
    start = old_slice.start + applied_slice.start * old_slice.step
    if start < 0:
        return slice(0)
    stop = old_slice.start + applied_slice.stop * old_slice.step
    if stop < 0:
        stop = None
    step = old_slice.step * applied_slice.step
    return slice(start, stop, step)