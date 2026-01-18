import pyomo.common.unittest as unittest
import pytest
import random
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentMap
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.convert import (
def test_convert_use_right(self):
    m = _make_model()
    time_points = [0.1, 0.2, 0.3, 0.4, 0.5]
    data = {m.var[:, 'A']: [1.0, 2.0, 3.0, 4.0, 5.0], m.var[:, 'B']: [6.0, 7.0, 8.0, 9.0, 10.0]}
    series_data = TimeSeriesData(data, time_points)
    interval_data = series_to_interval(series_data, use_left_endpoints=True)
    pred_data = IntervalData({m.var[:, 'A']: [1.0, 2.0, 3.0, 4.0], m.var[:, 'B']: [6.0, 7.0, 8.0, 9.0]}, [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)])
    self.assertEqual(pred_data, interval_data)