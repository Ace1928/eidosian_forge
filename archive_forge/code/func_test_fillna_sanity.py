import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
@pytest.mark.skipif(StorageFormat.get() == 'Hdk', reason="'datetime64[ns, pytz.FixedOffset(60)]' vs 'datetime64[ns, UTC+01:00]'")
def test_fillna_sanity():
    frame_data = [['a', 'a', np.nan, 'a'], ['b', 'b', np.nan, 'b'], ['c', 'c', np.nan, 'c']]
    df = pandas.DataFrame(frame_data)
    result = df.fillna({2: 'foo'})
    modin_df = pd.DataFrame(frame_data).fillna({2: 'foo'})
    df_equals(modin_df, result)
    modin_df = pd.DataFrame(df)
    df.fillna({2: 'foo'}, inplace=True)
    modin_df.fillna({2: 'foo'}, inplace=True)
    df_equals(modin_df, result)
    frame_data = {'Date': [pandas.NaT, pandas.Timestamp('2014-1-1')], 'Date2': [pandas.Timestamp('2013-1-1'), pandas.NaT]}
    df = pandas.DataFrame(frame_data)
    result = df.fillna(value={'Date': df['Date2']})
    modin_df = pd.DataFrame(frame_data).fillna(value={'Date': df['Date2']})
    df_equals(modin_df, result)
    frame_data = {'A': [pandas.Timestamp('2012-11-11 00:00:00+01:00'), pandas.NaT]}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    df_equals(modin_df.fillna(method='pad'), df.fillna(method='pad'))
    frame_data = {'A': [pandas.NaT, pandas.Timestamp('2012-11-11 00:00:00+01:00')]}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data).fillna(method='bfill')
    df_equals(modin_df, df.fillna(method='bfill'))