from __future__ import annotations
from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Generic, Literal, cast
import numpy as np
from numpy.typing import ArrayLike
from xarray.core import duck_array_ops, utils
from xarray.core.alignment import align, broadcast
from xarray.core.computation import apply_ufunc, dot
from xarray.core.types import Dims, T_DataArray, T_Xarray
from xarray.namedarray.utils import is_duck_dask_array
from xarray.util.deprecation_helpers import _deprecate_positional_args
@_deprecate_positional_args('v2023.10.0')
def sum_of_weights(self, dim: Dims=None, *, keep_attrs: bool | None=None) -> T_Xarray:
    return self._implementation(self._sum_of_weights, dim=dim, keep_attrs=keep_attrs)