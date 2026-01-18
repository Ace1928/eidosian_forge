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
def swap_dims(self, dims_dict: Mapping[Any, Hashable] | None=None, **dims_kwargs) -> Self:
    """Returns a new object with swapped dimensions.

        Parameters
        ----------
        dims_dict : dict-like
            Dictionary whose keys are current dimension names and whose values
            are new names.
        **dims_kwargs : {existing_dim: new_dim, ...}, optional
            The keyword arguments form of ``dims_dict``.
            One of dims_dict or dims_kwargs must be provided.

        Returns
        -------
        swapped : Dataset
            Dataset with swapped dimensions.

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     data_vars={"a": ("x", [5, 7]), "b": ("x", [0.1, 2.4])},
        ...     coords={"x": ["a", "b"], "y": ("x", [0, 1])},
        ... )
        >>> ds
        <xarray.Dataset> Size: 56B
        Dimensions:  (x: 2)
        Coordinates:
          * x        (x) <U1 8B 'a' 'b'
            y        (x) int64 16B 0 1
        Data variables:
            a        (x) int64 16B 5 7
            b        (x) float64 16B 0.1 2.4

        >>> ds.swap_dims({"x": "y"})
        <xarray.Dataset> Size: 56B
        Dimensions:  (y: 2)
        Coordinates:
            x        (y) <U1 8B 'a' 'b'
          * y        (y) int64 16B 0 1
        Data variables:
            a        (y) int64 16B 5 7
            b        (y) float64 16B 0.1 2.4

        >>> ds.swap_dims({"x": "z"})
        <xarray.Dataset> Size: 56B
        Dimensions:  (z: 2)
        Coordinates:
            x        (z) <U1 8B 'a' 'b'
            y        (z) int64 16B 0 1
        Dimensions without coordinates: z
        Data variables:
            a        (z) int64 16B 5 7
            b        (z) float64 16B 0.1 2.4

        See Also
        --------
        Dataset.rename
        DataArray.swap_dims
        """
    dims_dict = either_dict_or_kwargs(dims_dict, dims_kwargs, 'swap_dims')
    for current_name, new_name in dims_dict.items():
        if current_name not in self.dims:
            raise ValueError(f'cannot swap from dimension {current_name!r} because it is not one of the dimensions of this dataset {tuple(self.dims)}')
        if new_name in self.variables and self.variables[new_name].dims != (current_name,):
            raise ValueError(f'replacement dimension {new_name!r} is not a 1D variable along the old dimension {current_name!r}')
    result_dims = {dims_dict.get(dim, dim) for dim in self.dims}
    coord_names = self._coord_names.copy()
    coord_names.update({dim for dim in dims_dict.values() if dim in self.variables})
    variables: dict[Hashable, Variable] = {}
    indexes: dict[Hashable, Index] = {}
    for current_name, current_variable in self.variables.items():
        dims = tuple((dims_dict.get(dim, dim) for dim in current_variable.dims))
        var: Variable
        if current_name in result_dims:
            var = current_variable.to_index_variable()
            var.dims = dims
            if current_name in self._indexes:
                indexes[current_name] = self._indexes[current_name]
                variables[current_name] = var
            else:
                index, index_vars = create_default_index_implicit(var)
                indexes.update({name: index for name in index_vars})
                variables.update(index_vars)
                coord_names.update(index_vars)
        else:
            var = current_variable.to_base_variable()
            var.dims = dims
            variables[current_name] = var
    return self._replace_with_new_dims(variables, coord_names, indexes=indexes)