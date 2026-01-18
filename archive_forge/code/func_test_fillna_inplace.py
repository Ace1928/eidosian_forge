import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
def test_fillna_inplace():
    frame_data = random_state.randn(10, 4)
    df = pandas.DataFrame(frame_data)
    df[1][:4] = np.nan
    df[3][-4:] = np.nan
    modin_df = pd.DataFrame(df)
    df.fillna(value=0, inplace=True)
    try:
        df_equals(modin_df, df)
    except AssertionError:
        pass
    else:
        assert False
    modin_df.fillna(value=0, inplace=True)
    df_equals(modin_df, df)
    modin_df = pd.DataFrame(df).fillna(value={0: 0}, inplace=True)
    assert modin_df is None
    df[1][:4] = np.nan
    df[3][-4:] = np.nan
    modin_df = pd.DataFrame(df)
    df.fillna(method='ffill', inplace=True)
    try:
        df_equals(modin_df, df)
    except AssertionError:
        pass
    else:
        assert False
    modin_df.fillna(method='ffill', inplace=True)
    df_equals(modin_df, df)