import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.data.scalar_data import ScalarData
def test_contains_key(self):
    m = self._make_model()
    data = ScalarData({m.var[:, 'A']: 0.5}, time_set=m.time)
    self.assertTrue(data.contains_key(m.var[:, 'A']))
    self.assertFalse(data.contains_key(m.var[:, 'B']))