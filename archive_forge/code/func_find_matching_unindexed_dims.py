from __future__ import annotations
import functools
import operator
from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Final, Generic, TypeVar, cast, overload
import numpy as np
import pandas as pd
from xarray.core import dtypes
from xarray.core.indexes import (
from xarray.core.types import T_Alignable
from xarray.core.utils import is_dict_like, is_full_slice
from xarray.core.variable import Variable, as_compatible_data, calculate_dimensions
def find_matching_unindexed_dims(self) -> None:
    unindexed_dim_sizes = defaultdict(set)
    for obj in self.objects:
        for dim in obj.dims:
            if dim not in self.exclude_dims and dim not in obj.xindexes.dims:
                unindexed_dim_sizes[dim].add(obj.sizes[dim])
    self.unindexed_dim_sizes = unindexed_dim_sizes