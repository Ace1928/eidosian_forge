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
def make_empty(self, axes=None) -> Self:
    """return an empty BlockManager with the items axis of len 0"""
    if axes is None:
        axes = [Index([])] + self.axes[1:]
    if self.ndim == 1:
        assert isinstance(self, SingleBlockManager)
        blk = self.blocks[0]
        arr = blk.values[:0]
        bp = BlockPlacement(slice(0, 0))
        nb = blk.make_block_same_class(arr, placement=bp)
        blocks = [nb]
    else:
        blocks = []
    return type(self).from_blocks(blocks, axes)