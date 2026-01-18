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
def drop_isel(self, indexers=None, **indexers_kwargs) -> Self:
    """Drop index positions from this Dataset.

        Parameters
        ----------
        indexers : mapping of hashable to Any
            Index locations to drop
        **indexers_kwargs : {dim: position, ...}, optional
            The keyword arguments form of ``dim`` and ``positions``

        Returns
        -------
        dropped : Dataset

        Raises
        ------
        IndexError

        Examples
        --------
        >>> data = np.arange(6).reshape(2, 3)
        >>> labels = ["a", "b", "c"]
        >>> ds = xr.Dataset({"A": (["x", "y"], data), "y": labels})
        >>> ds
        <xarray.Dataset> Size: 60B
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * y        (y) <U1 12B 'a' 'b' 'c'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) int64 48B 0 1 2 3 4 5
        >>> ds.drop_isel(y=[0, 2])
        <xarray.Dataset> Size: 20B
        Dimensions:  (x: 2, y: 1)
        Coordinates:
          * y        (y) <U1 4B 'b'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) int64 16B 1 4
        >>> ds.drop_isel(y=1)
        <xarray.Dataset> Size: 40B
        Dimensions:  (x: 2, y: 2)
        Coordinates:
          * y        (y) <U1 8B 'a' 'c'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) int64 32B 0 2 3 5
        """
    indexers = either_dict_or_kwargs(indexers, indexers_kwargs, 'drop_isel')
    ds = self
    dimension_index = {}
    for dim, pos_for_dim in indexers.items():
        if utils.is_scalar(pos_for_dim):
            pos_for_dim = [pos_for_dim]
        pos_for_dim = np.asarray(pos_for_dim)
        index = self.get_index(dim)
        new_index = index.delete(pos_for_dim)
        dimension_index[dim] = new_index
    ds = ds.loc[dimension_index]
    return ds