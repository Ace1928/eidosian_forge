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
def reset_coords(self, names: Dims=None, drop: bool=False) -> Self:
    """Given names of coordinates, reset them to become variables

        Parameters
        ----------
        names : str, Iterable of Hashable or None, optional
            Name(s) of non-index coordinates in this dataset to reset into
            variables. By default, all non-index coordinates are reset.
        drop : bool, default: False
            If True, remove coordinates instead of converting them into
            variables.

        Examples
        --------
        >>> dataset = xr.Dataset(
        ...     {
        ...         "temperature": (
        ...             ["time", "lat", "lon"],
        ...             [[[25, 26], [27, 28]], [[29, 30], [31, 32]]],
        ...         ),
        ...         "precipitation": (
        ...             ["time", "lat", "lon"],
        ...             [[[0.5, 0.8], [0.2, 0.4]], [[0.3, 0.6], [0.7, 0.9]]],
        ...         ),
        ...     },
        ...     coords={
        ...         "time": pd.date_range(start="2023-01-01", periods=2),
        ...         "lat": [40, 41],
        ...         "lon": [-80, -79],
        ...         "altitude": 1000,
        ...     },
        ... )

        # Dataset before resetting coordinates

        >>> dataset
        <xarray.Dataset> Size: 184B
        Dimensions:        (time: 2, lat: 2, lon: 2)
        Coordinates:
          * time           (time) datetime64[ns] 16B 2023-01-01 2023-01-02
          * lat            (lat) int64 16B 40 41
          * lon            (lon) int64 16B -80 -79
            altitude       int64 8B 1000
        Data variables:
            temperature    (time, lat, lon) int64 64B 25 26 27 28 29 30 31 32
            precipitation  (time, lat, lon) float64 64B 0.5 0.8 0.2 0.4 0.3 0.6 0.7 0.9

        # Reset the 'altitude' coordinate

        >>> dataset_reset = dataset.reset_coords("altitude")

        # Dataset after resetting coordinates

        >>> dataset_reset
        <xarray.Dataset> Size: 184B
        Dimensions:        (time: 2, lat: 2, lon: 2)
        Coordinates:
          * time           (time) datetime64[ns] 16B 2023-01-01 2023-01-02
          * lat            (lat) int64 16B 40 41
          * lon            (lon) int64 16B -80 -79
        Data variables:
            temperature    (time, lat, lon) int64 64B 25 26 27 28 29 30 31 32
            precipitation  (time, lat, lon) float64 64B 0.5 0.8 0.2 0.4 0.3 0.6 0.7 0.9
            altitude       int64 8B 1000

        Returns
        -------
        Dataset

        See Also
        --------
        Dataset.set_coords
        """
    if names is None:
        names = self._coord_names - set(self._indexes)
    else:
        if isinstance(names, str) or not isinstance(names, Iterable):
            names = [names]
        else:
            names = list(names)
        self._assert_all_in_dataset(names)
        bad_coords = set(names) & set(self._indexes)
        if bad_coords:
            raise ValueError(f'cannot remove index coordinates with reset_coords: {bad_coords}')
    obj = self.copy()
    obj._coord_names.difference_update(names)
    if drop:
        for name in names:
            del obj._variables[name]
    return obj