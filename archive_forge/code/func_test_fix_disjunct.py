import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.opt import check_available_solvers
def test_fix_disjunct(self):
    """Test for deactivation of trivial constraints."""
    m = ConcreteModel()
    m.d1 = Disjunct()
    m.d2 = Disjunct()
    m.d2.c = Constraint()
    m.d = Disjunction(expr=[m.d1, m.d2])
    m.d1.indicator_var.set_value(True)
    m.d2.indicator_var.set_value(False)
    TransformationFactory('gdp.fix_disjuncts').apply_to(m)
    self.assertTrue(m.d1.indicator_var.fixed)
    self.assertTrue(m.d1.active)
    self.assertTrue(m.d2.indicator_var.fixed)
    self.assertFalse(m.d2.active)
    self.assertEqual(m.d1.ctype, Block)
    self.assertEqual(m.d2.ctype, Block)
    self.assertTrue(m.d2.c.active)