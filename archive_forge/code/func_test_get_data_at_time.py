import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
def test_get_data_at_time(self):
    m = self._make_model()
    data_dict = {m.var[:, 'A']: [1, 2, 3], m.var[:, 'B']: [2, 4, 6]}
    data = TimeSeriesData(data_dict, m.time)
    new_data = data.get_data_at_time(0.1)
    self.assertEqual(ScalarData({m.var[:, 'A']: 2, m.var[:, 'B']: 4}), new_data)
    t1 = 0.1
    new_data = data.get_data_at_time([t1])
    self.assertEqual(TimeSeriesData({m.var[:, 'A']: [2], m.var[:, 'B']: [4]}, [t1]), new_data)
    new_t = [0.0, 0.2]
    new_data = data.get_data_at_time(new_t)
    self.assertEqual(TimeSeriesData({m.var[:, 'A']: [1, 3], m.var[:, 'B']: [2, 6]}, new_t), new_data)