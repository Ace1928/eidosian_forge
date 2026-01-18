from __future__ import annotations
from collections.abc import (
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.tslibs import Timestamp
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import infer_dtype_from_scalar
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import (
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import (
from pandas.core.internals.base import (
from pandas.core.internals.blocks import (
from pandas.core.internals.ops import (
def reindex_indexer(self, new_axis: Index, indexer: npt.NDArray[np.intp] | None, axis: AxisInt, fill_value=None, allow_dups: bool=False, copy: bool | None=True, only_slice: bool=False, *, use_na_proxy: bool=False) -> Self:
    """
        Parameters
        ----------
        new_axis : Index
        indexer : ndarray[intp] or None
        axis : int
        fill_value : object, default None
        allow_dups : bool, default False
        copy : bool or None, default True
            If None, regard as False to get shallow copy.
        only_slice : bool, default False
            Whether to take views, not copies, along columns.
        use_na_proxy : bool, default False
            Whether to use a np.void ndarray for newly introduced columns.

        pandas-indexer with -1's only.
        """
    if copy is None:
        if using_copy_on_write():
            copy = False
        else:
            copy = True
    if indexer is None:
        if new_axis is self.axes[axis] and (not copy):
            return self
        result = self.copy(deep=copy)
        result.axes = list(self.axes)
        result.axes[axis] = new_axis
        return result
    assert isinstance(indexer, np.ndarray)
    if not allow_dups:
        self.axes[axis]._validate_can_reindex(indexer)
    if axis >= self.ndim:
        raise IndexError('Requested axis not found in manager')
    if axis == 0:
        new_blocks = self._slice_take_blocks_ax0(indexer, fill_value=fill_value, only_slice=only_slice, use_na_proxy=use_na_proxy)
    else:
        new_blocks = [blk.take_nd(indexer, axis=1, fill_value=fill_value if fill_value is not None else blk.fill_value) for blk in self.blocks]
    new_axes = list(self.axes)
    new_axes[axis] = new_axis
    new_mgr = type(self).from_blocks(new_blocks, new_axes)
    if axis == 1:
        new_mgr._blknos = self.blknos.copy()
        new_mgr._blklocs = self.blklocs.copy()
    return new_mgr