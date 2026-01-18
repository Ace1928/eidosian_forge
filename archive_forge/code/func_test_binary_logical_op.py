from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_binary_logical_op(self):

    def assert_eq(df, first, second, op, expected):
        if isinstance(first, str):
            first = df[first]
        else:
            first = Column(first)
        if isinstance(second, str):
            second = df[second]
        else:
            second = Column(second)
        res = self.to_pandas_series(self.op.binary_logical_op(first, second, op)).tolist()
        res = [True if x > 0 else False if x == 0 else None for x in res]
        assert_df_eq(pd.DataFrame({'a': res}), pd.DataFrame({'a': expected}))
    df = self.to_df([[True, True], [False, False], [True, False], [True, None], [False, None]], ['a', 'b'])
    assert_eq(df, 'b', True, 'and', [True, False, False, None, None])
    assert_eq(df, 'b', False, 'and', [False, False, False, False, False])
    assert_eq(df, 'b', True, 'or', [True, True, True, True, True])
    assert_eq(df, 'b', False, 'or', [True, False, False, None, None])
    assert_eq(df, 'b', 'a', 'and', [True, False, False, None, False])
    assert_eq(df, 'b', 'a', 'or', [True, False, True, True, None])