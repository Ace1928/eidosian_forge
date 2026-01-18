from __future__ import annotations
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence, Set
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union
import pandas as pd
from xarray.core import dtypes
from xarray.core.alignment import deep_align
from xarray.core.duck_array_ops import lazy_array_equiv
from xarray.core.indexes import (
from xarray.core.utils import Frozen, compat_dict_union, dict_equiv, equivalent
from xarray.core.variable import Variable, as_variable, calculate_dimensions
def dataset_update_method(dataset: Dataset, other: CoercibleMapping) -> _MergeResult:
    """Guts of the Dataset.update method.

    This drops a duplicated coordinates from `other` if `other` is not an
    `xarray.Dataset`, e.g., if it's a dict with DataArray values (GH2068,
    GH2180).
    """
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    if not isinstance(other, Dataset):
        other = dict(other)
        for key, value in other.items():
            if isinstance(value, DataArray):
                coord_names = [c for c in value.coords if c not in value.dims and c in dataset.coords]
                if coord_names:
                    other[key] = value.drop_vars(coord_names)
    return merge_core([dataset, other], priority_arg=1, indexes=dataset.xindexes, combine_attrs='override')