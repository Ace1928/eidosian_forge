from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
def test_combine_from_sequence_raises(monkeypatch):
    cls = DecimalArrayWithoutFromSequence

    @classmethod
    def construct_array_type(cls):
        return DecimalArrayWithoutFromSequence
    monkeypatch.setattr(DecimalDtype, 'construct_array_type', construct_array_type)
    arr = cls([decimal.Decimal('1.0'), decimal.Decimal('2.0')])
    ser = pd.Series(arr)
    result = ser.combine(ser, operator.add)
    expected = pd.Series([decimal.Decimal('2.0'), decimal.Decimal('4.0')], dtype='object')
    tm.assert_series_equal(result, expected)