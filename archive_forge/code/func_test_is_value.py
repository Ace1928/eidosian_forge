from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_is_value(self):

    def assert_eq(values, expr, positive, expected):
        col = self.op.to_col(values)
        res = self.to_pandas_series(self.op.is_value(col, IsValueSpec(expr, positive)))
        assert expected == res.tolist()
    assert_eq([1, 2, None], 'null', False, [True, True, False])
    assert_eq([1, 2, None], 'null', True, [False, False, True])
    assert_eq([True, False, None], 'true', True, [True, False, False])
    assert_eq([True, False, None], 'true', False, [False, True, True])
    assert_eq([True, False, None], 'false', True, [False, True, False])
    assert_eq([True, False, None], 'false', False, [True, False, True])