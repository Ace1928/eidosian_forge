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
def get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> Self:
    blk = self._block
    if using_copy_on_write() and len(indexer) > 0 and indexer.all():
        return type(self)(blk.copy(deep=False), self.index)
    array = blk.values[indexer]
    if isinstance(indexer, np.ndarray) and indexer.dtype.kind == 'b':
        refs = None
    else:
        refs = blk.refs
    bp = BlockPlacement(slice(0, len(array)))
    block = type(blk)(array, placement=bp, ndim=1, refs=refs)
    new_idx = self.index[indexer]
    return type(self)(block, new_idx)