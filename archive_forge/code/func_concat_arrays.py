from __future__ import annotations
import itertools
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.take import take_1d
from pandas.core.arrays import (
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.indexes.api import (
from pandas.core.indexes.base import get_values_for_csv
from pandas.core.internals.base import (
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import make_na_array
def concat_arrays(to_concat: list) -> ArrayLike:
    """
    Alternative for concat_compat but specialized for use in the ArrayManager.

    Differences: only deals with 1D arrays (no axis keyword), assumes
    ensure_wrapped_if_datetimelike and does not skip empty arrays to determine
    the dtype.
    In addition ensures that all NullArrayProxies get replaced with actual
    arrays.

    Parameters
    ----------
    to_concat : list of arrays

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    to_concat_no_proxy = [x for x in to_concat if not isinstance(x, NullArrayProxy)]
    dtypes = {x.dtype for x in to_concat_no_proxy}
    single_dtype = len(dtypes) == 1
    if single_dtype:
        target_dtype = to_concat_no_proxy[0].dtype
    elif all((lib.is_np_dtype(x, 'iub') for x in dtypes)):
        target_dtype = np_find_common_type(*dtypes)
    else:
        target_dtype = find_common_type([arr.dtype for arr in to_concat_no_proxy])
    to_concat = [arr.to_array(target_dtype) if isinstance(arr, NullArrayProxy) else astype_array(arr, target_dtype, copy=False) for arr in to_concat]
    if isinstance(to_concat[0], ExtensionArray):
        cls = type(to_concat[0])
        return cls._concat_same_type(to_concat)
    result = np.concatenate(to_concat)
    if len(result) == 0:
        kinds = {obj.dtype.kind for obj in to_concat_no_proxy}
        if len(kinds) != 1:
            if 'b' in kinds:
                result = result.astype(object)
    return result