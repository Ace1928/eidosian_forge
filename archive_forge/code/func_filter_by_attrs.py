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
def filter_by_attrs(self, **kwargs) -> Self:
    """Returns a ``Dataset`` with variables that match specific conditions.

        Can pass in ``key=value`` or ``key=callable``.  A Dataset is returned
        containing only the variables for which all the filter tests pass.
        These tests are either ``key=value`` for which the attribute ``key``
        has the exact value ``value`` or the callable passed into
        ``key=callable`` returns True. The callable will be passed a single
        value, either the value of the attribute ``key`` or ``None`` if the
        DataArray does not have an attribute with the name ``key``.

        Parameters
        ----------
        **kwargs
            key : str
                Attribute name.
            value : callable or obj
                If value is a callable, it should return a boolean in the form
                of bool = func(attr) where attr is da.attrs[key].
                Otherwise, value will be compared to the each
                DataArray's attrs[key].

        Returns
        -------
        new : Dataset
            New dataset with variables filtered by attribute.

        Examples
        --------
        >>> temp = 15 + 8 * np.random.randn(2, 2, 3)
        >>> precip = 10 * np.random.rand(2, 2, 3)
        >>> lon = [[-99.83, -99.32], [-99.79, -99.23]]
        >>> lat = [[42.25, 42.21], [42.63, 42.59]]
        >>> dims = ["x", "y", "time"]
        >>> temp_attr = dict(standard_name="air_potential_temperature")
        >>> precip_attr = dict(standard_name="convective_precipitation_flux")

        >>> ds = xr.Dataset(
        ...     dict(
        ...         temperature=(dims, temp, temp_attr),
        ...         precipitation=(dims, precip, precip_attr),
        ...     ),
        ...     coords=dict(
        ...         lon=(["x", "y"], lon),
        ...         lat=(["x", "y"], lat),
        ...         time=pd.date_range("2014-09-06", periods=3),
        ...         reference_time=pd.Timestamp("2014-09-05"),
        ...     ),
        ... )

        Get variables matching a specific standard_name:

        >>> ds.filter_by_attrs(standard_name="convective_precipitation_flux")
        <xarray.Dataset> Size: 192B
        Dimensions:         (x: 2, y: 2, time: 3)
        Coordinates:
            lon             (x, y) float64 32B -99.83 -99.32 -99.79 -99.23
            lat             (x, y) float64 32B 42.25 42.21 42.63 42.59
          * time            (time) datetime64[ns] 24B 2014-09-06 2014-09-07 2014-09-08
            reference_time  datetime64[ns] 8B 2014-09-05
        Dimensions without coordinates: x, y
        Data variables:
            precipitation   (x, y, time) float64 96B 5.68 9.256 0.7104 ... 4.615 7.805

        Get all variables that have a standard_name attribute:

        >>> standard_name = lambda v: v is not None
        >>> ds.filter_by_attrs(standard_name=standard_name)
        <xarray.Dataset> Size: 288B
        Dimensions:         (x: 2, y: 2, time: 3)
        Coordinates:
            lon             (x, y) float64 32B -99.83 -99.32 -99.79 -99.23
            lat             (x, y) float64 32B 42.25 42.21 42.63 42.59
          * time            (time) datetime64[ns] 24B 2014-09-06 2014-09-07 2014-09-08
            reference_time  datetime64[ns] 8B 2014-09-05
        Dimensions without coordinates: x, y
        Data variables:
            temperature     (x, y, time) float64 96B 29.11 18.2 22.83 ... 16.15 26.63
            precipitation   (x, y, time) float64 96B 5.68 9.256 0.7104 ... 4.615 7.805

        """
    selection = []
    for var_name, variable in self.variables.items():
        has_value_flag = False
        for attr_name, pattern in kwargs.items():
            attr_value = variable.attrs.get(attr_name)
            if callable(pattern) and pattern(attr_value) or attr_value == pattern:
                has_value_flag = True
            else:
                has_value_flag = False
                break
        if has_value_flag is True:
            selection.append(var_name)
    return self[selection]