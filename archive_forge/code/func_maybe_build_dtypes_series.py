import warnings
from typing import Optional
import numpy as np
import pandas
from pandas.api.types import is_bool_dtype, is_scalar
from modin.error_message import ErrorMessage
from .operator import Operator
def maybe_build_dtypes_series(first, second, dtype, trigger_computations=False) -> Optional[pandas.Series]:
    """
    Build a ``pandas.Series`` describing dtypes of the result of a binary operation.

    Parameters
    ----------
    first : PandasQueryCompiler
        First operand for which the binary operation would be performed later.
    second : PandasQueryCompiler, list-like or scalar
        Second operand for which the binary operation would be performed later.
    dtype : np.dtype
        Dtype of the result.
    trigger_computations : bool, default: False
        Whether to trigger computation of the lazy metadata for `first` and `second`.
        If False is specified this method will return None if any of the operands doesn't
        have materialized columns.

    Returns
    -------
    pandas.Series or None
        The pandas series with precomputed dtypes or None if there's not enough metadata to compute it.

    Notes
    -----
    Finds a union of columns and finds dtypes for all these columns.
    """
    if not trigger_computations:
        if not first._modin_frame.has_columns_cache:
            return None
        if isinstance(second, type(first)) and (not second._modin_frame.has_columns_cache):
            return None
    columns_first = set(first.columns)
    if isinstance(second, type(first)):
        columns_second = set(second.columns)
        columns_union = columns_first.union(columns_second)
    else:
        columns_union = columns_first
    dtypes = pandas.Series([dtype] * len(columns_union), index=columns_union)
    return dtypes