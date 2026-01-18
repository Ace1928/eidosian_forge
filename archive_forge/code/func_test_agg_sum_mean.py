from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_agg_sum_mean(self):

    def assert_eq(df, gp_keys, gp_map, expected, expected_cols):
        res = self.to_pandas_df(self.op.group_agg(df, gp_keys, gp_map))
        assert_df_eq(res, expected, expected_cols)
    pdf = self.to_df([[0, 1], [0, 1], [0, 2], [1, 5], [0, None]], ['a', 'b'])
    assert_eq(pdf, ['a'], {'b1': ('b', AggFunctionSpec('sum', unique=False, dropna=False)), 'b2': ('b', AggFunctionSpec('sum', unique=True, dropna=False)), 'b3': ('b', AggFunctionSpec('sum', unique=True, dropna=True)), 'b4': ('b', AggFunctionSpec('sum', unique=False, dropna=True))}, [[4, 3, 3, 4], [5, 5, 5, 5]], ['b1', 'b2', 'b3', 'b4'])
    pdf = self.to_df([[0, 1], [0, 1], [0, 4], [1, 5], [0, None]], ['a', 'b'])
    assert_eq(pdf, ['a'], {'b1': ('b', AggFunctionSpec('avg', unique=False, dropna=False)), 'b2': ('b', AggFunctionSpec('avg', unique=True, dropna=False)), 'b3': ('b', AggFunctionSpec('mean', unique=True, dropna=True)), 'b4': ('b', AggFunctionSpec('mean', unique=False, dropna=True))}, [[2, 2.5, 2.5, 2], [5, 5, 5, 5]], ['b1', 'b2', 'b3', 'b4'])
    assert_eq(pdf, ['a'], {'b1': ('b', AggFunctionSpec('sum', unique=True, dropna=False)), 'b2': ('b', AggFunctionSpec('avg', unique=True, dropna=False))}, [[5, 2.5], [5, 5]], ['b1', 'b2'])