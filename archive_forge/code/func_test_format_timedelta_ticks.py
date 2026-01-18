from datetime import (
import subprocess
import sys
import numpy as np
import pytest
import pandas._config.config as cf
from pandas._libs.tslibs import to_offset
from pandas import (
import pandas._testing as tm
from pandas.plotting import (
from pandas.tseries.offsets import (
@pytest.mark.parametrize('x, decimal, format_expected', [(0.0, 0, '00:00:00'), (3972320000000, 1, '01:06:12.3'), (713233432000000, 2, '8 days 06:07:13.43'), (32423432000000, 4, '09:00:23.4320')])
def test_format_timedelta_ticks(self, x, decimal, format_expected):
    tdc = converter.TimeSeries_TimedeltaFormatter
    result = tdc.format_timedelta_ticks(x, pos=None, n_decimals=decimal)
    assert result == format_expected