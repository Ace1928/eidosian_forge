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
@cache_readonly
def is_na(self) -> bool:
    blk = self.block
    if blk.dtype.kind == 'V':
        return True
    if not blk._can_hold_na:
        return False
    values = blk.values
    if values.size == 0:
        return True
    if isinstance(values.dtype, SparseDtype):
        return False
    if values.ndim == 1:
        val = values[0]
        if not is_scalar(val) or not isna(val):
            return False
        return isna_all(values)
    else:
        val = values[0][0]
        if not is_scalar(val) or not isna(val):
            return False
        return all((isna_all(row) for row in values))