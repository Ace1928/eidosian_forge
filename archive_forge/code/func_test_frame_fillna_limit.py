import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
@pytest.mark.parametrize('limit', [1, 2, 0.5, -1, -2, 1.5])
def test_frame_fillna_limit(data, limit):
    pandas_df = pandas.DataFrame(data)
    replace_pandas_series = pandas_df.columns.to_series().sample(frac=1)
    replace_dict = replace_pandas_series.to_dict()
    replace_pandas_df = pandas.DataFrame({col: pandas_df.index.to_series() for col in pandas_df.columns}, index=pandas_df.index).sample(frac=1)
    replace_modin_series = pd.Series(replace_pandas_series)
    replace_modin_df = pd.DataFrame(replace_pandas_df)
    index = pandas_df.index
    result = pandas_df[:2].reindex(index)
    modin_df = pd.DataFrame(result)
    if isinstance(limit, float):
        limit = int(len(modin_df) * limit)
    if limit is not None and limit < 0:
        limit = len(modin_df) + limit
    df_equals(modin_df.fillna(method='pad', limit=limit), result.fillna(method='pad', limit=limit))
    df_equals(modin_df.fillna(replace_dict, limit=limit), result.fillna(replace_dict, limit=limit))
    df_equals(modin_df.fillna(replace_modin_series, limit=limit), result.fillna(replace_pandas_series, limit=limit))
    df_equals(modin_df.fillna(replace_modin_df, limit=limit), result.fillna(replace_pandas_df, limit=limit))
    result = pandas_df[-2:].reindex(index)
    modin_df = pd.DataFrame(result)
    df_equals(modin_df.fillna(method='backfill', limit=limit), result.fillna(method='backfill', limit=limit))
    df_equals(modin_df.fillna(replace_dict, limit=limit), result.fillna(replace_dict, limit=limit))
    df_equals(modin_df.fillna(replace_modin_series, limit=limit), result.fillna(replace_pandas_series, limit=limit))
    df_equals(modin_df.fillna(replace_modin_df, limit=limit), result.fillna(replace_pandas_df, limit=limit))