import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
def test_get_time_points(self):
    m = self._make_model()
    data_dict = {m.var[:, 'A']: [1, 2, 3], m.var[:, 'B']: [2, 4, 6]}
    data = TimeSeriesData(data_dict, m.time)
    self.assertEqual(data.get_time_points(), list(m.time))
    new_time_list = [3, 4, 5]
    data = TimeSeriesData(data_dict, new_time_list)
    self.assertEqual(data.get_time_points(), new_time_list)