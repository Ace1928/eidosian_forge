from pyomo.common.errors import InfeasibleConstraintException
import pyomo.common.unittest as unittest
from pyomo.environ import Block, ConcreteModel, Constraint, TransformationFactory
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error
def test_fix_disjuncts_implied_by_true_disjunct(self):
    m = self.make_two_term_disjunction()
    m.d1.indicator_var.set_value(True)
    reverse = TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(m)
    self.check_fixed_mip(m)
    TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(m, reverse=reverse)
    self.assertFalse(m.d1.indicator_var.fixed)
    self.assertTrue(m.d1.active)
    self.assertTrue(m.d1.indicator_var.value)
    self.assertFalse(m.d2.indicator_var.fixed)
    self.assertTrue(m.d2.active)
    self.assertIsNone(m.d2.indicator_var.value)
    self.assertIs(m.d1.ctype, Disjunct)
    self.assertIs(m.d2.ctype, Disjunct)
    self.assertTrue(m.disjunction1.active)