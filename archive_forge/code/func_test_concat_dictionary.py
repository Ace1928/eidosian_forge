import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.utils import from_pandas
from modin.utils import get_current_execution
from .utils import (
@pytest.mark.parametrize('axis', [0, 1])
def test_concat_dictionary(axis):
    pandas_df, pandas_df2 = generate_dfs()
    modin_df, modin_df2 = (from_pandas(pandas_df), from_pandas(pandas_df2))
    df_equals(pd.concat({'A': modin_df, 'B': modin_df2}, axis=axis), pandas.concat({'A': pandas_df, 'B': pandas_df2}, axis=axis))