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
@property
def column_arrays(self) -> list[np.ndarray]:
    """
        Used in the JSON C code to access column arrays.
        This optimizes compared to using `iget_values` by converting each

        Warning! This doesn't handle Copy-on-Write, so should be used with
        caution (current use case of consuming this in the JSON code is fine).
        """
    result: list[np.ndarray | None] = [None] * len(self.items)
    for blk in self.blocks:
        mgr_locs = blk._mgr_locs
        values = blk.array_values._values_for_json()
        if values.ndim == 1:
            result[mgr_locs[0]] = values
        else:
            for i, loc in enumerate(mgr_locs):
                result[loc] = values[i]
    return result