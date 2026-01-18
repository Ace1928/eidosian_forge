import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', [None, 'Asia/Tokyo', 'US/Pacific'])
def test_delete2(self, tz):
    idx = date_range(start='2000-01-01 09:00', periods=10, freq='h', name='idx', tz=tz)
    expected = date_range(start='2000-01-01 10:00', periods=9, freq='h', name='idx', tz=tz)
    result = idx.delete(0)
    tm.assert_index_equal(result, expected)
    assert result.name == expected.name
    assert result.freqstr == 'h'
    assert result.tz == expected.tz
    expected = date_range(start='2000-01-01 09:00', periods=9, freq='h', name='idx', tz=tz)
    result = idx.delete(-1)
    tm.assert_index_equal(result, expected)
    assert result.name == expected.name
    assert result.freqstr == 'h'
    assert result.tz == expected.tz