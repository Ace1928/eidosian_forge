import warnings
from typing import Optional
import numpy as np
import pandas
from pandas.api.types import is_bool_dtype, is_scalar
from modin.error_message import ErrorMessage
from .operator import Operator
def maybe_compute_dtypes_common_cast(first, second, trigger_computations=False, axis=0, func=None) -> Optional[pandas.Series]:
    """
    Precompute data types for binary operations by finding common type between operands.

    Parameters
    ----------
    first : PandasQueryCompiler
        First operand for which the binary operation would be performed later.
    second : PandasQueryCompiler, list-like or scalar
        Second operand for which the binary operation would be performed later.
    trigger_computations : bool, default: False
        Whether to trigger computation of the lazy metadata for `first` and `second`.
        If False is specified this method will return None if any of the operands doesn't
        have materialized dtypes.
    axis : int, default: 0
        Axis to perform the binary operation along.
    func : callable(pandas.DataFrame, pandas.DataFrame) -> pandas.DataFrame, optional
        If specified, will use this function to perform the "try_sample" method
        (see ``Binary.register()`` docs for more details).

    Returns
    -------
    pandas.Series
        The pandas series with precomputed dtypes or None if there's not enough metadata to compute it.

    Notes
    -----
    The dtypes of the operands are supposed to be known.
    """
    if not trigger_computations:
        if not first._modin_frame.has_materialized_dtypes:
            return None
        if isinstance(second, type(first)) and (not second._modin_frame.has_materialized_dtypes):
            return None
    dtypes_first = first._modin_frame.dtypes.to_dict()
    if isinstance(second, type(first)):
        dtypes_second = second._modin_frame.dtypes.to_dict()
        columns_first = set(first.columns)
        columns_second = set(second.columns)
        common_columns = columns_first.intersection(columns_second)
        mismatch_columns = columns_first ^ columns_second
    elif isinstance(second, dict):
        dtypes_second = {key: pandas.api.types.pandas_dtype(type(value)) for key, value in second.items()}
        columns_first = set(first.columns)
        columns_second = set(second.keys())
        common_columns = columns_first.intersection(columns_second)
        mismatch_columns = columns_first.difference(columns_second)
    else:
        if isinstance(second, (list, tuple)):
            second_dtypes_list = [pandas.api.types.pandas_dtype(type(value)) for value in second] if axis == 1 else [np.array(second).dtype] * len(dtypes_first)
        elif is_scalar(second) or isinstance(second, np.ndarray):
            try:
                dtype = getattr(second, 'dtype', None) or pandas.api.types.pandas_dtype(type(second))
            except TypeError:
                dtype = pandas.Series(second).dtype
            second_dtypes_list = [dtype] * len(dtypes_first)
        else:
            raise NotImplementedError(f"Can't compute common type for {type(first)} and {type(second)}.")
        ErrorMessage.catch_bugs_and_request_email(failure_condition=len(second_dtypes_list) != len(dtypes_first), extra_log="Shapes of the operands of a binary operation don't match")
        dtypes_second = {key: second_dtypes_list[idx] for idx, key in enumerate(dtypes_first.keys())}
        common_columns = first.columns
        mismatch_columns = []
    nan_dtype = pandas.api.types.pandas_dtype(type(np.nan))
    dtypes = None
    if func is not None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                df1 = pandas.DataFrame([[1] * len(common_columns)]).astype({i: dtypes_first[col] for i, col in enumerate(common_columns)})
                df2 = pandas.DataFrame([[1] * len(common_columns)]).astype({i: dtypes_second[col] for i, col in enumerate(common_columns)})
                dtypes = func(df1, df2).dtypes.set_axis(common_columns)
        except TypeError:
            pass
    if dtypes is None:
        dtypes = pandas.Series([pandas.core.dtypes.cast.find_common_type([dtypes_first[x], dtypes_second[x]]) for x in common_columns], index=common_columns)
    dtypes = pandas.concat([dtypes, pandas.Series([nan_dtype] * len(mismatch_columns), index=mismatch_columns)])
    return dtypes