import re
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_asfreq_combined_pi(self):
    pi = PeriodIndex(['2001-01-01 00:00', '2001-01-02 02:00', 'NaT'], freq='h')
    exp = PeriodIndex(['2001-01-01 00:00', '2001-01-02 02:00', 'NaT'], freq='25h')
    for freq, how in zip(['1D1h', '1h1D'], ['S', 'E']):
        result = pi.asfreq(freq, how=how)
        tm.assert_index_equal(result, exp)
        assert result.freq == exp.freq
    for freq in ['1D1h', '1h1D']:
        pi = PeriodIndex(['2001-01-01 00:00', '2001-01-02 02:00', 'NaT'], freq=freq)
        result = pi.asfreq('h')
        exp = PeriodIndex(['2001-01-02 00:00', '2001-01-03 02:00', 'NaT'], freq='h')
        tm.assert_index_equal(result, exp)
        assert result.freq == exp.freq
        pi = PeriodIndex(['2001-01-01 00:00', '2001-01-02 02:00', 'NaT'], freq=freq)
        result = pi.asfreq('h', how='S')
        exp = PeriodIndex(['2001-01-01 00:00', '2001-01-02 02:00', 'NaT'], freq='h')
        tm.assert_index_equal(result, exp)
        assert result.freq == exp.freq