from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_comparison_op(self):

    def assert_eq(df, first, second, op, expected):
        if isinstance(first, str):
            first = df[first]
        else:
            first = Column(first)
        if isinstance(second, str):
            second = df[second]
        else:
            second = Column(second)
        self.assert_column_eq(self.op.comparison_op(first, second, op), expected)
    assert self.op.comparison_op(Column(0), Column(1), '<').native
    assert not self.op.comparison_op(Column(0), Column(1), '>').native
    df = self.to_df([[1, None], [1, 0], [1, 1], [1, 2]], ['a', 'b'])
    assert_eq(df, 'a', 0, '>', [True, True, True, True])
    assert_eq(df, 'a', 'b', '>', [None, True, False, False])
    assert_eq(df, 'a', 'b', '>=', [None, True, True, False])
    assert_eq(df, 'a', 'b', '==', [None, False, True, False])
    assert_eq(df, 'a', 'b', '!=', [None, True, False, True])
    assert_eq(df, 'a', 'b', '<=', [None, False, True, True])
    assert_eq(df, 'a', 'b', '<', [None, False, False, True])
    assert_eq(df, 0, 'a', '>', [False, False, False, False])
    assert_eq(df, 'b', 'a', '>', [None, False, False, True])
    assert_eq(df, 'b', 'a', '>=', [None, False, True, True])
    assert_eq(df, 'b', 'a', '==', [None, False, True, False])
    assert_eq(df, 'b', 'a', '!=', [None, True, False, True])
    assert_eq(df, 'b', 'a', '<=', [None, True, True, False])
    assert_eq(df, 'b', 'a', '<', [None, True, False, False])
    df = self.to_df([[1, None], [None, 1], [None, None]], ['a', 'b'])
    assert_eq(df, 'a', 'b', '>', [None, None, None])
    assert_eq(df, 'a', 'b', '==', [None, None, None])
    assert_eq(df, 'a', 'b', '!=', [None, None, None])
    df = self.to_df([['x', 'y'], ['x', None], ['x', 'x']], ['a', 'b'])
    assert_eq(df, 'a', 'b', '<', [True, None, False])
    assert_eq(df, 'a', 'b', '==', [False, None, True])
    assert_eq(df, 'a', 'b', '!=', [True, None, False])