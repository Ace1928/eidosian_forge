from __future__ import annotations
import collections.abc
import copy
from collections import defaultdict
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
import numpy as np
import pandas as pd
from xarray.core import formatting, nputils, utils
from xarray.core.indexing import (
from xarray.core.utils import (
def normalize_label(value, dtype=None) -> np.ndarray:
    if getattr(value, 'ndim', 1) <= 1:
        value = _asarray_tuplesafe(value)
    if dtype is not None and dtype.kind == 'f' and (value.dtype.kind != 'b'):
        value = np.asarray(value, dtype=dtype)
    return value