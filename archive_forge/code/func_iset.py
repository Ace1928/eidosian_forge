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
def iset(self, loc: int | slice | np.ndarray, value: ArrayLike, inplace: bool=False, refs: BlockValuesRefs | None=None) -> None:
    """
        Set new item in-place. Does not consolidate. Adds new Block if not
        contained in the current set of items
        """
    if self._blklocs is None and self.ndim > 1:
        self._rebuild_blknos_and_blklocs()
    value_is_extension_type = is_1d_only_ea_dtype(value.dtype)
    if not value_is_extension_type:
        if value.ndim == 2:
            value = value.T
        else:
            value = ensure_block_shape(value, ndim=2)
        if value.shape[1:] != self.shape[1:]:
            raise AssertionError('Shape of new values must be compatible with manager shape')
    if lib.is_integer(loc):
        loc = cast(int, loc)
        blkno = self.blknos[loc]
        blk = self.blocks[blkno]
        if len(blk._mgr_locs) == 1:
            return self._iset_single(loc, value, inplace=inplace, blkno=blkno, blk=blk, refs=refs)
        loc = [loc]
    if value_is_extension_type:

        def value_getitem(placement):
            return value
    else:

        def value_getitem(placement):
            return value[placement.indexer]
    blknos = self.blknos[loc]
    blklocs = self.blklocs[loc].copy()
    unfit_mgr_locs = []
    unfit_val_locs = []
    removed_blknos = []
    for blkno_l, val_locs in libinternals.get_blkno_placements(blknos, group=True):
        blk = self.blocks[blkno_l]
        blk_locs = blklocs[val_locs.indexer]
        if inplace and blk.should_store(value):
            if using_copy_on_write() and (not self._has_no_reference_block(blkno_l)):
                self._iset_split_block(blkno_l, blk_locs, value_getitem(val_locs), refs=refs)
            else:
                blk.set_inplace(blk_locs, value_getitem(val_locs))
                continue
        else:
            unfit_mgr_locs.append(blk.mgr_locs.as_array[blk_locs])
            unfit_val_locs.append(val_locs)
            if len(val_locs) == len(blk.mgr_locs):
                removed_blknos.append(blkno_l)
                continue
            else:
                self._iset_split_block(blkno_l, blk_locs, refs=refs)
    if len(removed_blknos):
        is_deleted = np.zeros(self.nblocks, dtype=np.bool_)
        is_deleted[removed_blknos] = True
        new_blknos = np.empty(self.nblocks, dtype=np.intp)
        new_blknos.fill(-1)
        new_blknos[~is_deleted] = np.arange(self.nblocks - len(removed_blknos))
        self._blknos = new_blknos[self._blknos]
        self.blocks = tuple((blk for i, blk in enumerate(self.blocks) if i not in set(removed_blknos)))
    if unfit_val_locs:
        unfit_idxr = np.concatenate(unfit_mgr_locs)
        unfit_count = len(unfit_idxr)
        new_blocks: list[Block] = []
        if value_is_extension_type:
            new_blocks.extend((new_block_2d(values=value, placement=BlockPlacement(slice(mgr_loc, mgr_loc + 1)), refs=refs) for mgr_loc in unfit_idxr))
            self._blknos[unfit_idxr] = np.arange(unfit_count) + len(self.blocks)
            self._blklocs[unfit_idxr] = 0
        else:
            unfit_val_items = unfit_val_locs[0].append(unfit_val_locs[1:])
            new_blocks.append(new_block_2d(values=value_getitem(unfit_val_items), placement=BlockPlacement(unfit_idxr), refs=refs))
            self._blknos[unfit_idxr] = len(self.blocks)
            self._blklocs[unfit_idxr] = np.arange(unfit_count)
        self.blocks += tuple(new_blocks)
        self._known_consolidated = False