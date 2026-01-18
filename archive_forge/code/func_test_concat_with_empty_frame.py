import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.utils import from_pandas
from modin.utils import get_current_execution
from .utils import (
def test_concat_with_empty_frame():
    modin_empty_df = pd.DataFrame()
    pandas_empty_df = pandas.DataFrame()
    modin_row = pd.Series({0: 'a', 1: 'b'})
    pandas_row = pandas.Series({0: 'a', 1: 'b'})
    df_equals(pd.concat([modin_empty_df, modin_row]), pandas.concat([pandas_empty_df, pandas_row]))
    md_empty1, pd_empty1 = create_test_dfs(index=[1, 2, 3])
    md_empty2, pd_empty2 = create_test_dfs(index=[2, 3, 4])
    df_equals(pd.concat([md_empty1, md_empty2], axis=0), pandas.concat([pd_empty1, pd_empty2], axis=0))
    df_equals(pd.concat([md_empty1, md_empty2], axis=1), pandas.concat([pd_empty1, pd_empty2], axis=1))