from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.generic import (
def concat_compat(to_concat: Sequence[ArrayLike], axis: AxisInt=0, ea_compat_axis: bool=False) -> ArrayLike:
    """
    provide concatenation of an array of arrays each of which is a single
    'normalized' dtypes (in that for example, if it's object, then it is a
    non-datetimelike and provide a combined dtype for the resulting array that
    preserves the overall dtype if possible)

    Parameters
    ----------
    to_concat : sequence of arrays
    axis : axis to provide concatenation
    ea_compat_axis : bool, default False
        For ExtensionArray compat, behave as if axis == 1 when determining
        whether to drop empty arrays.

    Returns
    -------
    a single array, preserving the combined dtypes
    """
    if len(to_concat) and lib.dtypes_all_equal([obj.dtype for obj in to_concat]):
        obj = to_concat[0]
        if isinstance(obj, np.ndarray):
            to_concat_arrs = cast('Sequence[np.ndarray]', to_concat)
            return np.concatenate(to_concat_arrs, axis=axis)
        to_concat_eas = cast('Sequence[ExtensionArray]', to_concat)
        if ea_compat_axis:
            return obj._concat_same_type(to_concat_eas)
        elif axis == 0:
            return obj._concat_same_type(to_concat_eas)
        else:
            return obj._concat_same_type(to_concat_eas, axis=axis)
    orig = to_concat
    non_empties = [x for x in to_concat if _is_nonempty(x, axis)]
    if non_empties and axis == 0 and (not ea_compat_axis):
        to_concat = non_empties
    any_ea, kinds, target_dtype = _get_result_dtype(to_concat, non_empties)
    if len(to_concat) < len(orig):
        _, _, alt_dtype = _get_result_dtype(orig, non_empties)
        if alt_dtype != target_dtype:
            warnings.warn('The behavior of array concatenation with empty entries is deprecated. In a future version, this will no longer exclude empty items when determining the result dtype. To retain the old behavior, exclude the empty entries before the concat operation.', FutureWarning, stacklevel=find_stack_level())
    if target_dtype is not None:
        to_concat = [astype_array(arr, target_dtype, copy=False) for arr in to_concat]
    if not isinstance(to_concat[0], np.ndarray):
        to_concat_eas = cast('Sequence[ExtensionArray]', to_concat)
        cls = type(to_concat[0])
        if ea_compat_axis or axis == 0:
            return cls._concat_same_type(to_concat_eas)
        else:
            return cls._concat_same_type(to_concat_eas, axis=axis)
    else:
        to_concat_arrs = cast('Sequence[np.ndarray]', to_concat)
        result = np.concatenate(to_concat_arrs, axis=axis)
        if not any_ea and 'b' in kinds and (result.dtype.kind in 'iuf'):
            result = result.astype(object, copy=False)
    return result