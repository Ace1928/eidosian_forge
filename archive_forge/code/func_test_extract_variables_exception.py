import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.data.scalar_data import ScalarData
def test_extract_variables_exception(self):
    m = self._make_model()
    data = ScalarData({m.var[:, 'A']: 0.5, m.var[:, 'B']: 2.0})
    msg = 'extract_variables with copy_values=True'
    with self.assertRaisesRegex(NotImplementedError, msg):
        data = data.extract_variables([m.var[:, 'A']], copy_values=True)