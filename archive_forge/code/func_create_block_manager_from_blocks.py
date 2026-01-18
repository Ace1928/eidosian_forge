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
def create_block_manager_from_blocks(blocks: list[Block], axes: list[Index], consolidate: bool=True, verify_integrity: bool=True) -> BlockManager:
    try:
        mgr = BlockManager(blocks, axes, verify_integrity=verify_integrity)
    except ValueError as err:
        arrays = [blk.values for blk in blocks]
        tot_items = sum((arr.shape[0] for arr in arrays))
        raise_construction_error(tot_items, arrays[0].shape[1:], axes, err)
    if consolidate:
        mgr._consolidate_inplace()
    return mgr