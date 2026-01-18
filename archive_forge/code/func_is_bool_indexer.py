from __future__ import annotations
import builtins
from collections import (
from collections.abc import (
import contextlib
from functools import partial
import inspect
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.compat.numpy import np_version_gte1p24
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import iterable_not_string
def is_bool_indexer(key: Any) -> bool:
    """
    Check whether `key` is a valid boolean indexer.

    Parameters
    ----------
    key : Any
        Only list-likes may be considered boolean indexers.
        All other types are not considered a boolean indexer.
        For array-like input, boolean ndarrays or ExtensionArrays
        with ``_is_boolean`` set are considered boolean indexers.

    Returns
    -------
    bool
        Whether `key` is a valid boolean indexer.

    Raises
    ------
    ValueError
        When the array is an object-dtype ndarray or ExtensionArray
        and contains missing values.

    See Also
    --------
    check_array_indexer : Check that `key` is a valid array to index,
        and convert to an ndarray.
    """
    if isinstance(key, (ABCSeries, np.ndarray, ABCIndex, ABCExtensionArray)) and (not isinstance(key, ABCMultiIndex)):
        if key.dtype == np.object_:
            key_array = np.asarray(key)
            if not lib.is_bool_array(key_array):
                na_msg = 'Cannot mask with non-boolean array containing NA / NaN values'
                if lib.is_bool_array(key_array, skipna=True):
                    raise ValueError(na_msg)
                return False
            return True
        elif is_bool_dtype(key.dtype):
            return True
    elif isinstance(key, list):
        if len(key) > 0:
            if type(key) is not list:
                key = list(key)
            return lib.is_bool_list(key)
    return False