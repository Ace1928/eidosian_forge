from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_except(self):

    def assert_eq(df1, df2, unique, expected, expected_cols):
        res = self.to_pandas_df(self.op.except_df(df1, df2, unique=unique))
        assert_df_eq(res, expected, expected_cols)
    a = self.to_df([['x', 'a'], ['x', 'a'], [None, None]], ['a', 'b'])
    b = self.to_df([['xx', 'aa'], [None, None], ['a', 'x']], ['b', 'a'])
    assert_eq(a, b, False, [['x', 'a'], ['x', 'a']], ['a', 'b'])
    assert_eq(a, b, True, [['x', 'a']], ['a', 'b'])
    b = self.to_df([['xx', 'aa'], [None, None], ['x', 'a']], ['b', 'a'])
    assert_eq(a, b, False, [], ['a', 'b'])
    assert_eq(a, b, True, [], ['a', 'b'])