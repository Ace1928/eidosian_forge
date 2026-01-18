from __future__ import annotations
from collections import abc
from typing import (
import numpy as np
from numpy import ma
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core import (
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
from pandas.core.indexes.api import (
from pandas.core.internals.array_manager import (
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import (
def convert_object_array(content: list[npt.NDArray[np.object_]], dtype: DtypeObj | None, dtype_backend: str='numpy', coerce_float: bool=False) -> list[ArrayLike]:
    """
    Internal function to convert object array.

    Parameters
    ----------
    content: List[np.ndarray]
    dtype: np.dtype or ExtensionDtype
    dtype_backend: Controls if nullable/pyarrow dtypes are returned.
    coerce_float: Cast floats that are integers to int.

    Returns
    -------
    List[ArrayLike]
    """

    def convert(arr):
        if dtype != np.dtype('O'):
            arr = lib.maybe_convert_objects(arr, try_float=coerce_float, convert_to_nullable_dtype=dtype_backend != 'numpy')
            if dtype is None:
                if arr.dtype == np.dtype('O'):
                    arr = maybe_infer_to_datetimelike(arr)
                    if dtype_backend != 'numpy' and arr.dtype == np.dtype('O'):
                        new_dtype = StringDtype()
                        arr_cls = new_dtype.construct_array_type()
                        arr = arr_cls._from_sequence(arr, dtype=new_dtype)
                elif dtype_backend != 'numpy' and isinstance(arr, np.ndarray):
                    if arr.dtype.kind in 'iufb':
                        arr = pd_array(arr, copy=False)
            elif isinstance(dtype, ExtensionDtype):
                cls = dtype.construct_array_type()
                arr = cls._from_sequence(arr, dtype=dtype, copy=False)
            elif dtype.kind in 'mM':
                arr = maybe_cast_to_datetime(arr, dtype)
        return arr
    arrays = [convert(arr) for arr in content]
    return arrays