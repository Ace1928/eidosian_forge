import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.data.scalar_data import ScalarData
def test_construct_and_get_data(self):
    m = self._make_model()
    data = ScalarData({m.var[:, 'A']: 0.5, m.var[:, 'B']: 2.0})
    data_dict = data.get_data()
    pred_data_dict = {pyo.ComponentUID('var[*,A]'): 0.5, pyo.ComponentUID('var[*,B]'): 2.0}
    self.assertEqual(data_dict, pred_data_dict)