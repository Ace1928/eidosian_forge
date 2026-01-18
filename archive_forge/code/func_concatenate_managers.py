from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.missing import NA
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.internals.array_manager import ArrayManager
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import (
def concatenate_managers(mgrs_indexers, axes: list[Index], concat_axis: AxisInt, copy: bool) -> Manager2D:
    """
    Concatenate block managers into one.

    Parameters
    ----------
    mgrs_indexers : list of (BlockManager, {axis: indexer,...}) tuples
    axes : list of Index
    concat_axis : int
    copy : bool

    Returns
    -------
    BlockManager
    """
    needs_copy = copy and concat_axis == 0
    if isinstance(mgrs_indexers[0][0], ArrayManager):
        mgrs = _maybe_reindex_columns_na_proxy(axes, mgrs_indexers, needs_copy)
        return _concatenate_array_managers(mgrs, axes, concat_axis)
    if concat_axis == 0:
        mgrs = _maybe_reindex_columns_na_proxy(axes, mgrs_indexers, needs_copy)
        return mgrs[0].concat_horizontal(mgrs, axes)
    if len(mgrs_indexers) > 0 and mgrs_indexers[0][0].nblocks > 0:
        first_dtype = mgrs_indexers[0][0].blocks[0].dtype
        if first_dtype in [np.float64, np.float32]:
            if all((_is_homogeneous_mgr(mgr, first_dtype) for mgr, _ in mgrs_indexers)) and len(mgrs_indexers) > 1:
                shape = tuple((len(x) for x in axes))
                nb = _concat_homogeneous_fastpath(mgrs_indexers, shape, first_dtype)
                return BlockManager((nb,), axes)
    mgrs = _maybe_reindex_columns_na_proxy(axes, mgrs_indexers, needs_copy)
    if len(mgrs) == 1:
        mgr = mgrs[0]
        out = mgr.copy(deep=False)
        out.axes = axes
        return out
    concat_plan = _get_combined_plan(mgrs)
    blocks = []
    values: ArrayLike
    for placement, join_units in concat_plan:
        unit = join_units[0]
        blk = unit.block
        if _is_uniform_join_units(join_units):
            vals = [ju.block.values for ju in join_units]
            if not blk.is_extension:
                values = np.concatenate(vals, axis=1)
            elif is_1d_only_ea_dtype(blk.dtype):
                values = concat_compat(vals, axis=0, ea_compat_axis=True)
                values = ensure_block_shape(values, ndim=2)
            else:
                values = concat_compat(vals, axis=1)
            values = ensure_wrapped_if_datetimelike(values)
            fastpath = blk.values.dtype == values.dtype
        else:
            values = _concatenate_join_units(join_units, copy=copy)
            fastpath = False
        if fastpath:
            b = blk.make_block_same_class(values, placement=placement)
        else:
            b = new_block_2d(values, placement=placement)
        blocks.append(b)
    return BlockManager(tuple(blocks), axes)