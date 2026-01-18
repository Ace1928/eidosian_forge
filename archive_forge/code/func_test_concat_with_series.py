import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.utils import from_pandas
from modin.utils import get_current_execution
from .utils import (
def test_concat_with_series():
    df, df2 = generate_dfs()
    modin_df, modin_df2 = (from_pandas(df), from_pandas(df2))
    pandas_series = pandas.Series([1, 2, 3, 4], name='new_col')
    df_equals(pd.concat([modin_df, modin_df2, pandas_series], axis=0), pandas.concat([df, df2, pandas_series], axis=0))
    df_equals(pd.concat([modin_df, modin_df2, pandas_series], axis=1), pandas.concat([df, df2, pandas_series], axis=1))