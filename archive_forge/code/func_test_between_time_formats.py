from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import timezones
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@td.skip_if_not_us_locale
def test_between_time_formats(self, frame_or_series):
    rng = date_range('1/1/2000', '1/5/2000', freq='5min')
    ts = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 2)), index=rng)
    ts = tm.get_obj(ts, frame_or_series)
    strings = [('2:00', '2:30'), ('0200', '0230'), ('2:00am', '2:30am'), ('0200am', '0230am'), ('2:00:00', '2:30:00'), ('020000', '023000'), ('2:00:00am', '2:30:00am'), ('020000am', '023000am')]
    expected_length = 28
    for time_string in strings:
        assert len(ts.between_time(*time_string)) == expected_length