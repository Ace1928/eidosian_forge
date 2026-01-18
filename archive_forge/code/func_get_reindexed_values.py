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
def get_reindexed_values(self, empty_dtype: DtypeObj, upcasted_na) -> ArrayLike:
    values: ArrayLike
    if upcasted_na is None and self.block.dtype.kind != 'V':
        return self.block.values
    else:
        fill_value = upcasted_na
        if self._is_valid_na_for(empty_dtype):
            blk_dtype = self.block.dtype
            if blk_dtype == np.dtype('object'):
                values = cast(np.ndarray, self.block.values)
                if values.size and values[0, 0] is None:
                    fill_value = None
            return make_na_array(empty_dtype, self.block.shape, fill_value)
        return self.block.values