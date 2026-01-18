import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', [None, 'Asia/Tokyo', 'US/Pacific'])
def test_delete_slice2(self, tz, unit):
    dti = date_range('2000-01-01 09:00', periods=10, freq='h', name='idx', tz=tz, unit=unit)
    ts = Series(1, index=dti)
    result = ts.drop(ts.index[:5]).index
    expected = dti[5:]
    tm.assert_index_equal(result, expected)
    assert result.name == expected.name
    assert result.freq == expected.freq
    assert result.tz == expected.tz
    result = ts.drop(ts.index[[1, 3, 5, 7, 9]]).index
    expected = dti[::2]._with_freq(None)
    tm.assert_index_equal(result, expected)
    assert result.name == expected.name
    assert result.freq == expected.freq
    assert result.tz == expected.tz