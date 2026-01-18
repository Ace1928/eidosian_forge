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
def column_setitem(self, loc: int, idx: int | slice | np.ndarray, value, inplace_only: bool=False) -> None:
    """
        Set values ("setitem") into a single column (not setting the full column).

        This is a method on the BlockManager level, to avoid creating an
        intermediate Series at the DataFrame level (`s = df[loc]; s[idx] = value`)
        """
    needs_to_warn = False
    if warn_copy_on_write() and (not self._has_no_reference(loc)):
        if not isinstance(self.blocks[self.blknos[loc]].values, (ArrowExtensionArray, ArrowStringArray)):
            needs_to_warn = True
    elif using_copy_on_write() and (not self._has_no_reference(loc)):
        blkno = self.blknos[loc]
        blk_loc = self.blklocs[loc]
        values = self.blocks[blkno].values
        if values.ndim == 1:
            values = values.copy()
        else:
            values = values[[blk_loc]]
        self._iset_split_block(blkno, [blk_loc], values)
    col_mgr = self.iget(loc, track_ref=False)
    if inplace_only:
        col_mgr.setitem_inplace(idx, value)
    else:
        new_mgr = col_mgr.setitem((idx,), value)
        self.iset(loc, new_mgr._block.values, inplace=True)
    if needs_to_warn:
        warnings.warn(COW_WARNING_GENERAL_MSG, FutureWarning, stacklevel=find_stack_level())