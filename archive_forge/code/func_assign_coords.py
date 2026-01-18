from __future__ import annotations
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from contextlib import suppress
from html import escape
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes, duck_array_ops, formatting, formatting_html, ops
from xarray.core.indexing import BasicIndexer, ExplicitlyIndexed
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import (
from xarray.namedarray.core import _raise_if_any_duplicate_dimensions
from xarray.namedarray.parallelcompat import get_chunked_array_type, guess_chunkmanager
from xarray.namedarray.pycompat import is_chunked_array
def assign_coords(self, coords: Mapping | None=None, **coords_kwargs: Any) -> Self:
    """Assign new coordinates to this object.

        Returns a new object with all the original data in addition to the new
        coordinates.

        Parameters
        ----------
        coords : mapping of dim to coord, optional
            A mapping whose keys are the names of the coordinates and values are the
            coordinates to assign. The mapping will generally be a dict or
            :class:`Coordinates`.

            * If a value is a standard data value — for example, a ``DataArray``,
              scalar, or array — the data is simply assigned as a coordinate.

            * If a value is callable, it is called with this object as the only
              parameter, and the return value is used as new coordinate variables.

            * A coordinate can also be defined and attached to an existing dimension
              using a tuple with the first element the dimension name and the second
              element the values for this new coordinate.

        **coords_kwargs : optional
            The keyword arguments form of ``coords``.
            One of ``coords`` or ``coords_kwargs`` must be provided.

        Returns
        -------
        assigned : same type as caller
            A new object with the new coordinates in addition to the existing
            data.

        Examples
        --------
        Convert `DataArray` longitude coordinates from 0-359 to -180-179:

        >>> da = xr.DataArray(
        ...     np.random.rand(4),
        ...     coords=[np.array([358, 359, 0, 1])],
        ...     dims="lon",
        ... )
        >>> da
        <xarray.DataArray (lon: 4)> Size: 32B
        array([0.5488135 , 0.71518937, 0.60276338, 0.54488318])
        Coordinates:
          * lon      (lon) int64 32B 358 359 0 1
        >>> da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
        <xarray.DataArray (lon: 4)> Size: 32B
        array([0.5488135 , 0.71518937, 0.60276338, 0.54488318])
        Coordinates:
          * lon      (lon) int64 32B -2 -1 0 1

        The function also accepts dictionary arguments:

        >>> da.assign_coords({"lon": (((da.lon + 180) % 360) - 180)})
        <xarray.DataArray (lon: 4)> Size: 32B
        array([0.5488135 , 0.71518937, 0.60276338, 0.54488318])
        Coordinates:
          * lon      (lon) int64 32B -2 -1 0 1

        New coordinate can also be attached to an existing dimension:

        >>> lon_2 = np.array([300, 289, 0, 1])
        >>> da.assign_coords(lon_2=("lon", lon_2))
        <xarray.DataArray (lon: 4)> Size: 32B
        array([0.5488135 , 0.71518937, 0.60276338, 0.54488318])
        Coordinates:
          * lon      (lon) int64 32B 358 359 0 1
            lon_2    (lon) int64 32B 300 289 0 1

        Note that the same result can also be obtained with a dict e.g.

        >>> _ = da.assign_coords({"lon_2": ("lon", lon_2)})

        Note the same method applies to `Dataset` objects.

        Convert `Dataset` longitude coordinates from 0-359 to -180-179:

        >>> temperature = np.linspace(20, 32, num=16).reshape(2, 2, 4)
        >>> precipitation = 2 * np.identity(4).reshape(2, 2, 4)
        >>> ds = xr.Dataset(
        ...     data_vars=dict(
        ...         temperature=(["x", "y", "time"], temperature),
        ...         precipitation=(["x", "y", "time"], precipitation),
        ...     ),
        ...     coords=dict(
        ...         lon=(["x", "y"], [[260.17, 260.68], [260.21, 260.77]]),
        ...         lat=(["x", "y"], [[42.25, 42.21], [42.63, 42.59]]),
        ...         time=pd.date_range("2014-09-06", periods=4),
        ...         reference_time=pd.Timestamp("2014-09-05"),
        ...     ),
        ...     attrs=dict(description="Weather-related data"),
        ... )
        >>> ds
        <xarray.Dataset> Size: 360B
        Dimensions:         (x: 2, y: 2, time: 4)
        Coordinates:
            lon             (x, y) float64 32B 260.2 260.7 260.2 260.8
            lat             (x, y) float64 32B 42.25 42.21 42.63 42.59
          * time            (time) datetime64[ns] 32B 2014-09-06 ... 2014-09-09
            reference_time  datetime64[ns] 8B 2014-09-05
        Dimensions without coordinates: x, y
        Data variables:
            temperature     (x, y, time) float64 128B 20.0 20.8 21.6 ... 30.4 31.2 32.0
            precipitation   (x, y, time) float64 128B 2.0 0.0 0.0 0.0 ... 0.0 0.0 2.0
        Attributes:
            description:  Weather-related data
        >>> ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        <xarray.Dataset> Size: 360B
        Dimensions:         (x: 2, y: 2, time: 4)
        Coordinates:
            lon             (x, y) float64 32B -99.83 -99.32 -99.79 -99.23
            lat             (x, y) float64 32B 42.25 42.21 42.63 42.59
          * time            (time) datetime64[ns] 32B 2014-09-06 ... 2014-09-09
            reference_time  datetime64[ns] 8B 2014-09-05
        Dimensions without coordinates: x, y
        Data variables:
            temperature     (x, y, time) float64 128B 20.0 20.8 21.6 ... 30.4 31.2 32.0
            precipitation   (x, y, time) float64 128B 2.0 0.0 0.0 0.0 ... 0.0 0.0 2.0
        Attributes:
            description:  Weather-related data

        See Also
        --------
        Dataset.assign
        Dataset.swap_dims
        Dataset.set_coords
        """
    from xarray.core.coordinates import Coordinates
    coords_combined = either_dict_or_kwargs(coords, coords_kwargs, 'assign_coords')
    data = self.copy(deep=False)
    results: Coordinates | dict[Hashable, Any]
    if isinstance(coords, Coordinates):
        results = coords
    else:
        results = self._calc_assign_results(coords_combined)
    data.coords.update(results)
    return data