import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
def test_get_cuid_context(self):
    m = self._make_model()
    top = pyo.ConcreteModel()
    top.m = m
    pred_cuid = pyo.ComponentUID(m.var[:, 'A'], context=m)
    self.assertEqual(get_indexed_cuid(m.var[:, 'A'], context=m), pred_cuid)
    self.assertEqual(get_indexed_cuid(pyo.Reference(m.var[:, 'A']), context=m), pred_cuid)
    full_cuid = pyo.ComponentUID(m.var[:, 'A'])
    self.assertNotEqual(get_indexed_cuid('m.var[*,A]'), pred_cuid)
    self.assertEqual(get_indexed_cuid('m.var[*,A]'), full_cuid)
    msg = 'Context is not allowed'
    with self.assertRaisesRegex(ValueError, msg):
        get_indexed_cuid('m.var[*,A]', context=m)
    self.assertEqual(get_indexed_cuid(m.var[0, 'A'], sets=(m.time,), context=m), pred_cuid)