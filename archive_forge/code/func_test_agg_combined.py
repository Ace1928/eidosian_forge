from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_agg_combined(self):

    def assert_eq(df, gp_keys, gp_map, expected, expected_cols):
        res = self.to_pandas_df(self.op.group_agg(df, gp_keys, gp_map))
        assert_df_eq(res, expected, expected_cols)
    pdf = self.to_df([[0, 1, None], [0, 1, None], [0, 4, 1], [1, None, None], [None, None, None]], ['a', 'b', 'c'])
    assert_eq(pdf, ['a'], {'a1': ('a', AggFunctionSpec('sum', unique=False, dropna=False)), 'a2': ('*', AggFunctionSpec('count', unique=True, dropna=False)), 'a3': ('c', AggFunctionSpec('sum', unique=True, dropna=False)), 'a4': ('*', AggFunctionSpec('count', unique=False, dropna=False)), 'a5': ('c', AggFunctionSpec('min', unique=False, dropna=False)), 'a6': ('b,c', AggFunctionSpec('count', unique=True, dropna=False))}, [[0.0, 2.0, 1, 3.0, 1.0, 2.0], [1.0, 1.0, None, 1.0, None, 1.0], [None, 1.0, None, 1.0, None, 1.0]], ['a1', 'a2', 'a3', 'a4', 'a5', 'a6'])