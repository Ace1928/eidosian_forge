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
@pytest.mark.parametrize('view_interval', [(1, 2), (2, 1)])
def test_call_w_different_view_intervals(self, view_interval, monkeypatch):

    class mock_axis:

        def get_view_interval(self):
            return view_interval
    tdc = converter.TimeSeries_TimedeltaFormatter()
    monkeypatch.setattr(tdc, 'axis', mock_axis())
    tdc(0.0, 0)