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
def test_as_arrow(self):
    df = self.df([], 'a:int,b:int')
    assert [] == list(ArrowDataFrame(fi.as_arrow(df)).as_dict_iterable())
    assert fi.is_local(fi.as_arrow(df))
    df = self.df([[pd.NaT, 1]], 'a:datetime,b:int')
    assert [dict(a=None, b=1)] == list(ArrowDataFrame(fi.as_arrow(df)).as_dict_iterable())
    df = self.df([[pd.Timestamp('2020-01-01'), 1]], 'a:datetime,b:int')
    assert [dict(a=datetime(2020, 1, 1), b=1)] == list(ArrowDataFrame(fi.as_arrow(df)).as_dict_iterable())
    data = [[[float('nan'), 2.0]]]
    df = self.df(data, 'a:[float]')
    assert [[[None, 2.0]]] == ArrowDataFrame(fi.as_arrow(df)).as_array()
    data = [[dict(b=True)]]
    df = self.df(data, 'a:{b:bool}')
    assert data == ArrowDataFrame(fi.as_arrow(df)).as_array()
    data = [[[dict(b=[30, 40])]]]
    df = self.df(data, 'a:[{b:[long]}]')
    assert data == ArrowDataFrame(fi.as_arrow(df)).as_array()