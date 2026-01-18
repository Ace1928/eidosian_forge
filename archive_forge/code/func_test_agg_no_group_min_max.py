from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_agg_no_group_min_max(self):

    def assert_eq(df, gp_map, expected, expected_cols):
        res = self.to_pandas_df(self.op.group_agg(df, [], gp_map))
        assert_df_eq(res, expected, expected_cols)
    pdf = self.to_df([[0, 1], [0, 1], [1, None], [1, 5], [0, None]], ['a', 'b'])
    assert_eq(pdf, {'b1': ('b', AggFunctionSpec('min', unique=False, dropna=False)), 'b2': ('b', AggFunctionSpec('min', unique=True, dropna=False)), 'b3': ('b', AggFunctionSpec('min', unique=True, dropna=True)), 'b4': ('b', AggFunctionSpec('min', unique=False, dropna=True))}, [[1.0, 1.0, 1.0, 1.0]], ['b1', 'b2', 'b3', 'b4'])
    assert_eq(pdf, {'b1': ('b', AggFunctionSpec('max', unique=False, dropna=False)), 'b2': ('b', AggFunctionSpec('max', unique=True, dropna=False)), 'b3': ('b', AggFunctionSpec('max', unique=True, dropna=True)), 'b4': ('b', AggFunctionSpec('max', unique=False, dropna=True))}, [[5, 5, 5, 5]], ['b1', 'b2', 'b3', 'b4'])