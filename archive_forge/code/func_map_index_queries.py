from __future__ import annotations
import enum
import functools
import operator
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import timedelta
from html import escape
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd
from xarray.core import duck_array_ops
from xarray.core.nputils import NumpyVIndexAdapter
from xarray.core.options import OPTIONS
from xarray.core.types import T_Xarray
from xarray.core.utils import (
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import array_type, integer_types, is_chunked_array
def map_index_queries(obj: T_Xarray, indexers: Mapping[Any, Any], method=None, tolerance: int | float | Iterable[int | float] | None=None, **indexers_kwargs: Any) -> IndexSelResult:
    """Execute index queries from a DataArray / Dataset and label-based indexers
    and return the (merged) query results.

    """
    from xarray.core.dataarray import DataArray
    if method is None and tolerance is None:
        options = {}
    else:
        options = {'method': method, 'tolerance': tolerance}
    indexers = either_dict_or_kwargs(indexers, indexers_kwargs, 'map_index_queries')
    grouped_indexers = group_indexers_by_index(obj, indexers, options)
    results = []
    for index, labels in grouped_indexers:
        if index is None:
            results.append(IndexSelResult(labels))
        else:
            results.append(index.sel(labels, **options))
    merged = merge_sel_results(results)
    for k, v in merged.dim_indexers.items():
        if isinstance(v, DataArray):
            if k in v._indexes:
                v = v.reset_index(k)
            drop_coords = [name for name in v._coords if name in merged.dim_indexers]
            merged.dim_indexers[k] = v.drop_vars(drop_coords)
    return merged