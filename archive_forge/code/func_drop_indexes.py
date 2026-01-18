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
def drop_indexes(self, coord_names: Hashable | Iterable[Hashable], *, errors: ErrorOptions='raise') -> Self:
    """Drop the indexes assigned to the given coordinates.

        Parameters
        ----------
        coord_names : hashable or iterable of hashable
            Name(s) of the coordinate(s) for which to drop the index.
        errors : {"raise", "ignore"}, default: "raise"
            If 'raise', raises a ValueError error if any of the coordinates
            passed have no index or are not in the dataset.
            If 'ignore', no error is raised.

        Returns
        -------
        dropped : Dataset
            A new dataset with dropped indexes.

        """
    if is_scalar(coord_names) or not isinstance(coord_names, Iterable):
        coord_names = {coord_names}
    else:
        coord_names = set(coord_names)
    if errors == 'raise':
        invalid_coords = coord_names - self._coord_names
        if invalid_coords:
            raise ValueError(f'The coordinates {tuple(invalid_coords)} are not found in the dataset coordinates {tuple(self.coords.keys())}')
        unindexed_coords = set(coord_names) - set(self._indexes)
        if unindexed_coords:
            raise ValueError(f'those coordinates do not have an index: {unindexed_coords}')
    assert_no_index_corrupted(self.xindexes, coord_names, action='remove index(es)')
    variables = {}
    for name, var in self._variables.items():
        if name in coord_names:
            variables[name] = var.to_base_variable()
        else:
            variables[name] = var
    indexes = {k: v for k, v in self._indexes.items() if k not in coord_names}
    return self._replace(variables=variables, indexes=indexes)