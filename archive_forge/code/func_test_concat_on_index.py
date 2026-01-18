import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.utils import from_pandas
from modin.utils import get_current_execution
from .utils import (
def test_concat_on_index():
    df, df2 = generate_dfs()
    modin_df, modin_df2 = (from_pandas(df), from_pandas(df2))
    df_equals(pd.concat([modin_df, modin_df2], axis='index'), pandas.concat([df, df2], axis='index'))
    df_equals(pd.concat([modin_df, modin_df2], axis='rows'), pandas.concat([df, df2], axis='rows'))
    df_equals(pd.concat([modin_df, modin_df2], axis=0), pandas.concat([df, df2], axis=0))