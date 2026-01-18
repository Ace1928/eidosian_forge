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
def interpolate_na(self, dim: Hashable | None=None, method: InterpOptions='linear', limit: int | None=None, use_coordinate: bool | Hashable=True, max_gap: int | float | str | pd.Timedelta | np.timedelta64 | datetime.timedelta=None, **kwargs: Any) -> Self:
    """Fill in NaNs by interpolating according to different methods.

        Parameters
        ----------
        dim : Hashable or None, optional
            Specifies the dimension along which to interpolate.
        method : {"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial",             "barycentric", "krogh", "pchip", "spline", "akima"}, default: "linear"
            String indicating which method to use for interpolation:

            - 'linear': linear interpolation. Additional keyword
              arguments are passed to :py:func:`numpy.interp`
            - 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial':
              are passed to :py:func:`scipy.interpolate.interp1d`. If
              ``method='polynomial'``, the ``order`` keyword argument must also be
              provided.
            - 'barycentric', 'krogh', 'pchip', 'spline', 'akima': use their
              respective :py:class:`scipy.interpolate` classes.

        use_coordinate : bool or Hashable, default: True
            Specifies which index to use as the x values in the interpolation
            formulated as `y = f(x)`. If False, values are treated as if
            equally-spaced along ``dim``. If True, the IndexVariable `dim` is
            used. If ``use_coordinate`` is a string, it specifies the name of a
            coordinate variable to use as the index.
        limit : int, default: None
            Maximum number of consecutive NaNs to fill. Must be greater than 0
            or None for no limit. This filling is done regardless of the size of
            the gap in the data. To only interpolate over gaps less than a given length,
            see ``max_gap``.
        max_gap : int, float, str, pandas.Timedelta, numpy.timedelta64, datetime.timedelta, default: None
            Maximum size of gap, a continuous sequence of NaNs, that will be filled.
            Use None for no limit. When interpolating along a datetime64 dimension
            and ``use_coordinate=True``, ``max_gap`` can be one of the following:

            - a string that is valid input for pandas.to_timedelta
            - a :py:class:`numpy.timedelta64` object
            - a :py:class:`pandas.Timedelta` object
            - a :py:class:`datetime.timedelta` object

            Otherwise, ``max_gap`` must be an int or a float. Use of ``max_gap`` with unlabeled
            dimensions has not been implemented yet. Gap length is defined as the difference
            between coordinate values at the first data point after a gap and the last value
            before a gap. For gaps at the beginning (end), gap length is defined as the difference
            between coordinate values at the first (last) valid data point and the first (last) NaN.
            For example, consider::

                <xarray.DataArray (x: 9)>
                array([nan, nan, nan,  1., nan, nan,  4., nan, nan])
                Coordinates:
                  * x        (x) int64 0 1 2 3 4 5 6 7 8

            The gap lengths are 3-0 = 3; 6-3 = 3; and 8-6 = 2 respectively
        **kwargs : dict, optional
            parameters passed verbatim to the underlying interpolation function

        Returns
        -------
        interpolated: Dataset
            Filled in Dataset.

        Warning
        --------
        When passing fill_value as a keyword argument with method="linear", it does not use
        ``numpy.interp`` but it uses ``scipy.interpolate.interp1d``, which provides the fill_value parameter.

        See Also
        --------
        numpy.interp
        scipy.interpolate

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     {
        ...         "A": ("x", [np.nan, 2, 3, np.nan, 0]),
        ...         "B": ("x", [3, 4, np.nan, 1, 7]),
        ...         "C": ("x", [np.nan, np.nan, np.nan, 5, 0]),
        ...         "D": ("x", [np.nan, 3, np.nan, -1, 4]),
        ...     },
        ...     coords={"x": [0, 1, 2, 3, 4]},
        ... )
        >>> ds
        <xarray.Dataset> Size: 200B
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 40B 0 1 2 3 4
        Data variables:
            A        (x) float64 40B nan 2.0 3.0 nan 0.0
            B        (x) float64 40B 3.0 4.0 nan 1.0 7.0
            C        (x) float64 40B nan nan nan 5.0 0.0
            D        (x) float64 40B nan 3.0 nan -1.0 4.0

        >>> ds.interpolate_na(dim="x", method="linear")
        <xarray.Dataset> Size: 200B
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 40B 0 1 2 3 4
        Data variables:
            A        (x) float64 40B nan 2.0 3.0 1.5 0.0
            B        (x) float64 40B 3.0 4.0 2.5 1.0 7.0
            C        (x) float64 40B nan nan nan 5.0 0.0
            D        (x) float64 40B nan 3.0 1.0 -1.0 4.0

        >>> ds.interpolate_na(dim="x", method="linear", fill_value="extrapolate")
        <xarray.Dataset> Size: 200B
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 40B 0 1 2 3 4
        Data variables:
            A        (x) float64 40B 1.0 2.0 3.0 1.5 0.0
            B        (x) float64 40B 3.0 4.0 2.5 1.0 7.0
            C        (x) float64 40B 20.0 15.0 10.0 5.0 0.0
            D        (x) float64 40B 5.0 3.0 1.0 -1.0 4.0
        """
    from xarray.core.missing import _apply_over_vars_with_dim, interp_na
    new = _apply_over_vars_with_dim(interp_na, self, dim=dim, method=method, limit=limit, use_coordinate=use_coordinate, max_gap=max_gap, **kwargs)
    return new