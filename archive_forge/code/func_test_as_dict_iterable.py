from datetime import date, datetime
from typing import Any
from unittest import TestCase
import numpy as np
import pandas as pd
from pytest import raises
import fugue.api as fi
from fugue.dataframe import ArrowDataFrame, DataFrame
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.exceptions import FugueDataFrameOperationError, FugueDatasetEmptyError
def test_as_dict_iterable(self):
    df = self.df([[pd.NaT, 1]], 'a:datetime,b:int')
    assert [dict(a=None, b=1)] == list(fi.as_dict_iterable(df))
    df = self.df([[pd.NaT, 1]], 'a:datetime,b:int')
    assert [dict(b=1)] == list(fi.as_dict_iterable(df, ['b']))
    df = self.df([[pd.Timestamp('2020-01-01'), 1]], 'a:datetime,b:int')
    assert [dict(a=datetime(2020, 1, 1), b=1)] == list(fi.as_dict_iterable(df))
    df = self.df([[pd.Timestamp('2020-01-01'), 1]], 'a:datetime,b:int')
    assert [dict(b=1)] == list(fi.as_dict_iterable(df, ['b']))