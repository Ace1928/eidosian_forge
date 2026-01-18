import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_astype_object2(self):
    idx = period_range(start='2013-01-01', periods=4, freq='M', name='idx')
    expected_list = [Period('2013-01-31', freq='M'), Period('2013-02-28', freq='M'), Period('2013-03-31', freq='M'), Period('2013-04-30', freq='M')]
    expected = Index(expected_list, dtype=object, name='idx')
    result = idx.astype(object)
    assert isinstance(result, Index)
    assert result.dtype == object
    tm.assert_index_equal(result, expected)
    assert result.name == expected.name
    assert idx.tolist() == expected_list
    idx = PeriodIndex(['2013-01-01', '2013-01-02', 'NaT', '2013-01-04'], freq='D', name='idx')
    expected_list = [Period('2013-01-01', freq='D'), Period('2013-01-02', freq='D'), Period('NaT', freq='D'), Period('2013-01-04', freq='D')]
    expected = Index(expected_list, dtype=object, name='idx')
    result = idx.astype(object)
    assert isinstance(result, Index)
    assert result.dtype == object
    tm.assert_index_equal(result, expected)
    for i in [0, 1, 3]:
        assert result[i] == expected[i]
    assert result[2] is NaT
    assert result.name == expected.name
    result_list = idx.tolist()
    for i in [0, 1, 3]:
        assert result_list[i] == expected_list[i]
    assert result_list[2] is NaT