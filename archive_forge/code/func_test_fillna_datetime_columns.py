import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
def test_fillna_datetime_columns():
    frame_data = {'A': [-1, -2, np.nan], 'B': pd.date_range('20130101', periods=3), 'C': ['foo', 'bar', None], 'D': ['foo2', 'bar2', None]}
    df = pandas.DataFrame(frame_data, index=pd.date_range('20130110', periods=3))
    modin_df = pd.DataFrame(frame_data, index=pd.date_range('20130110', periods=3))
    df_equals(modin_df.fillna('?'), df.fillna('?'))
    frame_data = {'A': [-1, -2, np.nan], 'B': [pandas.Timestamp('2013-01-01'), pandas.Timestamp('2013-01-02'), pandas.NaT], 'C': ['foo', 'bar', None], 'D': ['foo2', 'bar2', None]}
    df = pandas.DataFrame(frame_data, index=pd.date_range('20130110', periods=3))
    modin_df = pd.DataFrame(frame_data, index=pd.date_range('20130110', periods=3))
    df_equals(modin_df.fillna('?'), df.fillna('?'))