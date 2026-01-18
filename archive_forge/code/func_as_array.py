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
def as_array(self, dtype: np.dtype | None=None, copy: bool=False, na_value: object=lib.no_default) -> np.ndarray:
    """
        Convert the blockmanager data into an numpy array.

        Parameters
        ----------
        dtype : np.dtype or None, default None
            Data type of the return array.
        copy : bool, default False
            If True then guarantee that a copy is returned. A value of
            False does not guarantee that the underlying data is not
            copied.
        na_value : object, default lib.no_default
            Value to be used as the missing value sentinel.

        Returns
        -------
        arr : ndarray
        """
    passed_nan = lib.is_float(na_value) and isna(na_value)
    if len(self.blocks) == 0:
        arr = np.empty(self.shape, dtype=float)
        return arr.transpose()
    if self.is_single_block:
        blk = self.blocks[0]
        if na_value is not lib.no_default:
            if lib.is_np_dtype(blk.dtype, 'f') and passed_nan:
                pass
            else:
                copy = True
        if blk.is_extension:
            arr = blk.values.to_numpy(dtype=dtype, na_value=na_value, copy=copy).reshape(blk.shape)
        else:
            arr = np.array(blk.values, dtype=dtype, copy=copy)
        if using_copy_on_write() and (not copy):
            arr = arr.view()
            arr.flags.writeable = False
    else:
        arr = self._interleave(dtype=dtype, na_value=na_value)
    if na_value is lib.no_default:
        pass
    elif arr.dtype.kind == 'f' and passed_nan:
        pass
    else:
        arr[isna(arr)] = na_value
    return arr.transpose()