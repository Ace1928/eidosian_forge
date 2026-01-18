from __future__ import annotations
import copy
import datetime
import inspect
import itertools
import math
import sys
import warnings
from collections import defaultdict
from collections.abc import (
from html import escape
from numbers import Number
from operator import methodcaller
from os import PathLike
from typing import IO, TYPE_CHECKING, Any, Callable, Generic, Literal, cast, overload
import numpy as np
import pandas as pd
from xarray.coding.calendar_ops import convert_calendar, interp_calendar
from xarray.coding.cftimeindex import CFTimeIndex, _parse_array_of_cftime_strings
from xarray.core import (
from xarray.core import dtypes as xrdtypes
from xarray.core._aggregations import DatasetAggregations
from xarray.core.alignment import (
from xarray.core.arithmetic import DatasetArithmetic
from xarray.core.common import (
from xarray.core.computation import unify_chunks
from xarray.core.coordinates import (
from xarray.core.duck_array_ops import datetime_to_numeric
from xarray.core.indexes import (
from xarray.core.indexing import is_fancy_indexer, map_index_queries
from xarray.core.merge import (
from xarray.core.missing import get_clean_interp_index
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import (
from xarray.core.utils import (
from xarray.core.variable import (
from xarray.namedarray.parallelcompat import get_chunked_array_type, guess_chunkmanager
from xarray.namedarray.pycompat import array_type, is_chunked_array
from xarray.plot.accessor import DatasetPlotAccessor
from xarray.util.deprecation_helpers import _deprecate_positional_args
def broadcast_like(self, other: T_DataArrayOrSet, exclude: Iterable[Hashable] | None=None) -> Self:
    """Broadcast this DataArray against another Dataset or DataArray.
        This is equivalent to xr.broadcast(other, self)[1]

        Parameters
        ----------
        other : Dataset or DataArray
            Object against which to broadcast this array.
        exclude : iterable of hashable, optional
            Dimensions that must not be broadcasted

        """
    if exclude is None:
        exclude = set()
    else:
        exclude = set(exclude)
    args = align(other, self, join='outer', copy=False, exclude=exclude)
    dims_map, common_coords = _get_broadcast_dims_map_common_coords(args, exclude)
    return _broadcast_helper(args[1], exclude, dims_map, common_coords)