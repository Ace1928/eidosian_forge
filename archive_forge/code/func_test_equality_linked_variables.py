import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.preprocessing.plugins.var_aggregator import (
from pyomo.environ import (
def test_equality_linked_variables(self):
    """Test for equality-linked variable detection."""
    m = self.build_model()
    self.assertEqual(_get_equality_linked_variables(m.c1), ())
    self.assertEqual(_get_equality_linked_variables(m.c2), ())
    c3 = _get_equality_linked_variables(m.c3)
    self.assertIn(m.v3, ComponentSet(c3))
    self.assertIn(m.v4, ComponentSet(c3))
    self.assertEqual(len(c3), 2)
    self.assertEqual(_get_equality_linked_variables(m.ignore_me), ())
    self.assertEqual(_get_equality_linked_variables(m.ignore_me_too), ())
    self.assertEqual(_get_equality_linked_variables(m.multiple), ())