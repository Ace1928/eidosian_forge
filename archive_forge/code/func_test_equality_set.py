import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.preprocessing.plugins.var_aggregator import (
from pyomo.environ import (
def test_equality_set(self):
    """Test for equality set map generation."""
    m = self.build_model()
    eq_var_map = _build_equality_set(m)
    self.assertIsNone(eq_var_map.get(m.z1, None))
    self.assertIsNone(eq_var_map.get(m.v1, None))
    self.assertIsNone(eq_var_map.get(m.v2, None))
    self.assertEqual(eq_var_map[m.v3], ComponentSet([m.v3, m.v4]))
    self.assertEqual(eq_var_map[m.v4], ComponentSet([m.v3, m.v4]))
    self.assertEqual(eq_var_map[m.x[1]], ComponentSet([m.x[1], m.x[2], m.x[3], m.x[4]]))
    self.assertEqual(eq_var_map[m.x[2]], ComponentSet([m.x[1], m.x[2], m.x[3], m.x[4]]))
    self.assertEqual(eq_var_map[m.x[3]], ComponentSet([m.x[1], m.x[2], m.x[3], m.x[4]]))
    self.assertEqual(eq_var_map[m.x[4]], ComponentSet([m.x[1], m.x[2], m.x[3], m.x[4]]))
    self.assertEqual(eq_var_map[m.y[1]], ComponentSet([m.y[1], m.y[2]]))
    self.assertEqual(eq_var_map[m.y[2]], ComponentSet([m.y[1], m.y[2]]))