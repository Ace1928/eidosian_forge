from __future__ import annotations
import datetime
from functools import partial
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import roperator
from pandas.core.computation import expressions
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.ops import missing
from pandas.core.ops.dispatch import should_extension_dispatch
from pandas.core.ops.invalid import invalid_comparison
def logical_op(left: ArrayLike, right: Any, op) -> ArrayLike:
    """
    Evaluate a logical operation `|`, `&`, or `^`.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
    right : object
        Cannot be a DataFrame, Series, or Index.
    op : {operator.and_, operator.or_, operator.xor}
        Or one of the reversed variants from roperator.

    Returns
    -------
    ndarray or ExtensionArray
    """

    def fill_bool(x, left=None):
        if x.dtype.kind in 'cfO':
            mask = isna(x)
            if mask.any():
                x = x.astype(object)
                x[mask] = False
        if left is None or left.dtype.kind == 'b':
            x = x.astype(bool)
        return x
    right = lib.item_from_zerodim(right)
    if is_list_like(right) and (not hasattr(right, 'dtype')):
        warnings.warn('Logical ops (and, or, xor) between Pandas objects and dtype-less sequences (e.g. list, tuple) are deprecated and will raise in a future version. Wrap the object in a Series, Index, or np.array before operating instead.', FutureWarning, stacklevel=find_stack_level())
        right = construct_1d_object_array_from_listlike(right)
    lvalues = ensure_wrapped_if_datetimelike(left)
    rvalues = right
    if should_extension_dispatch(lvalues, rvalues):
        res_values = op(lvalues, rvalues)
    else:
        if isinstance(rvalues, np.ndarray):
            is_other_int_dtype = rvalues.dtype.kind in 'iu'
            if not is_other_int_dtype:
                rvalues = fill_bool(rvalues, lvalues)
        else:
            is_other_int_dtype = lib.is_integer(rvalues)
        res_values = na_logical_op(lvalues, rvalues, op)
        if not (left.dtype.kind in 'iu' and is_other_int_dtype):
            res_values = fill_bool(res_values)
    return res_values