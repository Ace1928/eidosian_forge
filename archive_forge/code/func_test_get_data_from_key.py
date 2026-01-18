import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.data.scalar_data import ScalarData
def test_get_data_from_key(self):
    m = self._make_model()
    data = ScalarData({m.var[:, 'A']: 0.5, m.var[:, 'B']: 2.0}, time_set=m.time)
    val = data.get_data_from_key(m.var[:, 'A'])
    self.assertEqual(val, 0.5)
    val = data.get_data_from_key(pyo.Reference(m.var[:, 'A']))
    self.assertEqual(val, 0.5)
    val = data.get_data_from_key(m.var[0, 'A'])
    self.assertEqual(val, 0.5)
    val = data.get_data_from_key('var[*,A]')
    self.assertEqual(val, 0.5)