import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
def test_fillna_dataframe():
    frame_data = {'a': [np.nan, 1, 2, np.nan, np.nan], 'b': [1, 2, 3, np.nan, np.nan], 'c': [np.nan, 1, 2, 3, 4]}
    df = pandas.DataFrame(frame_data, index=list('VWXYZ'))
    modin_df = pd.DataFrame(frame_data, index=list('VWXYZ'))
    df2 = pandas.DataFrame({'a': [np.nan, 10, 20, 30, 40], 'b': [50, 60, 70, 80, 90], 'foo': ['bar'] * 5}, index=list('VWXuZ'))
    modin_df2 = pd.DataFrame(df2)
    df_equals(modin_df.fillna(modin_df2), df.fillna(df2))