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
def drop_sel(self, labels=None, *, errors: ErrorOptions='raise', **labels_kwargs) -> Self:
    """Drop index labels from this dataset.

        Parameters
        ----------
        labels : mapping of hashable to Any
            Index labels to drop
        errors : {"raise", "ignore"}, default: "raise"
            If 'raise', raises a ValueError error if
            any of the index labels passed are not
            in the dataset. If 'ignore', any given labels that are in the
            dataset are dropped and no error is raised.
        **labels_kwargs : {dim: label, ...}, optional
            The keyword arguments form of ``dim`` and ``labels``

        Returns
        -------
        dropped : Dataset

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
        >>> ds.drop_sel(y=["a", "c"])
        <xarray.Dataset> Size: 20B
        Dimensions:  (x: 2, y: 1)
        Coordinates:
          * y        (y) <U1 4B 'b'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) int64 16B 1 4
        >>> ds.drop_sel(y="b")
        <xarray.Dataset> Size: 40B
        Dimensions:  (x: 2, y: 2)
        Coordinates:
          * y        (y) <U1 8B 'a' 'c'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) int64 32B 0 2 3 5
        """
    if errors not in ['raise', 'ignore']:
        raise ValueError('errors must be either "raise" or "ignore"')
    labels = either_dict_or_kwargs(labels, labels_kwargs, 'drop_sel')
    ds = self
    for dim, labels_for_dim in labels.items():
        if utils.is_scalar(labels_for_dim):
            labels_for_dim = [labels_for_dim]
        labels_for_dim = np.asarray(labels_for_dim)
        try:
            index = self.get_index(dim)
        except KeyError:
            raise ValueError(f'dimension {dim!r} does not have coordinate labels')
        new_index = index.drop(labels_for_dim, errors=errors)
        ds = ds.loc[{dim: new_index}]
    return ds