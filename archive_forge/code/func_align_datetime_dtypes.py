import datetime
import pandas
import pytest
from pandas.core.dtypes.common import is_datetime64_any_dtype, is_object_dtype
import modin.pandas as pd
from modin.tests.pandas.utils import df_equals
from modin.tests.pandas.utils import eval_io as general_eval_io
from modin.utils import try_cast_to_pandas
def align_datetime_dtypes(*dfs):
    """
    Make all of the passed frames have DateTime dtype for the same columns.

    Cast column type of the certain frame to the DateTime type if any frame in
    the `dfs` sequence has DateTime type for this column.

    Parameters
    ----------
    *dfs : iterable of DataFrames
        DataFrames to align DateTime dtypes.

    Notes
    -----
    Passed Modin frames may be casted to pandas in the result.
    """
    datetime_cols = {}
    time_cols = set()
    for df in dfs:
        for col, dtype in df.dtypes.items():
            if col not in datetime_cols and is_datetime64_any_dtype(dtype):
                datetime_cols[col] = dtype
            elif dtype == pandas.api.types.pandas_dtype('O') and col not in time_cols and (len(df) > 0) and all((isinstance(val, datetime.time) or pandas.isna(val) for val in df[col])):
                time_cols.add(col)
    if len(datetime_cols) == 0 and len(time_cols) == 0:
        return dfs

    def convert_to_time(value):
        """Convert passed value to `datetime.time`."""
        if isinstance(value, datetime.time):
            return value
        elif isinstance(value, str):
            return datetime.time.fromisoformat(value)
        else:
            return datetime.time(value)
    time_cols_list = list(time_cols)
    casted_dfs = []
    for df in dfs:
        pandas_df = try_cast_to_pandas(df)
        if datetime_cols:
            pandas_df = pandas_df.astype(datetime_cols)
        if time_cols:
            pandas_df[time_cols_list] = pandas_df[time_cols_list].map(convert_to_time)
        casted_dfs.append(pandas_df)
    return casted_dfs