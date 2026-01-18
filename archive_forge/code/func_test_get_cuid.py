import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
def test_get_cuid(self):
    m = self._make_model()
    pred_cuid = pyo.ComponentUID(m.var[:, 'A'])
    self.assertEqual(get_indexed_cuid(m.var[:, 'A']), pred_cuid)
    self.assertEqual(get_indexed_cuid(pyo.Reference(m.var[:, 'A'])), pred_cuid)
    self.assertEqual(get_indexed_cuid('var[*,A]'), pred_cuid)
    self.assertEqual(get_indexed_cuid("var[*,'A']"), pred_cuid)
    self.assertEqual(get_indexed_cuid(m.var[0, 'A'], sets=(m.time,)), pred_cuid)