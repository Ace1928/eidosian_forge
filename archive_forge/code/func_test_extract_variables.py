import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.data.scalar_data import ScalarData
def test_extract_variables(self):
    m = self._make_model()
    data = ScalarData({m.var[:, 'A']: 0.5, m.var[:, 'B']: 2.0})
    data = data.extract_variables([m.var[:, 'A']])
    pred_data_dict = {pyo.ComponentUID(m.var[:, 'A']): 0.5}
    self.assertEqual(data.get_data(), pred_data_dict)