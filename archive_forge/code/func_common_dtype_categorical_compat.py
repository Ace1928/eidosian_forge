from __future__ import annotations
import datetime as dt
import functools
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import (
from pandas._libs.missing import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.compat.numpy import np_version_gt2
from pandas.errors import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
from pandas.core.dtypes.missing import (
from pandas.io._util import _arrow_dtype_mapping
def common_dtype_categorical_compat(objs: Sequence[Index | ArrayLike], dtype: DtypeObj) -> DtypeObj:
    """
    Update the result of find_common_type to account for NAs in a Categorical.

    Parameters
    ----------
    objs : list[np.ndarray | ExtensionArray | Index]
    dtype : np.dtype or ExtensionDtype

    Returns
    -------
    np.dtype or ExtensionDtype
    """
    if lib.is_np_dtype(dtype, 'iu'):
        for obj in objs:
            obj_dtype = getattr(obj, 'dtype', None)
            if isinstance(obj_dtype, CategoricalDtype):
                if isinstance(obj, ABCIndex):
                    hasnas = obj.hasnans
                else:
                    hasnas = cast('Categorical', obj)._hasna
                if hasnas:
                    dtype = np.dtype(np.float64)
                    break
    return dtype