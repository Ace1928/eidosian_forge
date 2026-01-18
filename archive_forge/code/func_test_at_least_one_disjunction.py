from pyomo.common.errors import InfeasibleConstraintException
import pyomo.common.unittest as unittest
from pyomo.environ import Block, ConcreteModel, Constraint, TransformationFactory
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error
def test_at_least_one_disjunction(self):
    m = ConcreteModel()
    self.add_three_term_disjunction(m, exactly_one=False)
    m.d[1].indicator_var.fix(True)
    m.d[2].indicator_var = True
    m.d[3].indicator_var = False
    reverse = TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(m)
    self.assertTrue(m.d[1].indicator_var.fixed)
    self.assertTrue(m.d[1].indicator_var.value)
    self.assertTrue(m.d[1].active)
    self.assertIs(m.d[1].ctype, Block)
    self.assertTrue(m.d[2].indicator_var.fixed)
    self.assertTrue(m.d[2].indicator_var.value)
    self.assertTrue(m.d[2].active)
    self.assertIs(m.d[2].ctype, Block)
    self.assertTrue(m.d[3].indicator_var.fixed)
    self.assertFalse(m.d[3].indicator_var.value)
    self.assertFalse(m.d[3].active)
    self.assertIs(m.d[3].ctype, Block)
    self.assertFalse(m.disjunction2.active)
    reverse = TransformationFactory('gdp.transform_current_disjunctive_state').apply_to(m, reverse=reverse)
    self.assertTrue(m.d[1].indicator_var.fixed)
    self.assertTrue(m.d[1].indicator_var.value)
    self.assertTrue(m.d[1].active)
    self.assertIs(m.d[1].ctype, Disjunct)
    self.assertFalse(m.d[2].indicator_var.fixed)
    self.assertTrue(m.d[2].indicator_var.value)
    self.assertTrue(m.d[2].active)
    self.assertIs(m.d[2].ctype, Disjunct)
    self.assertFalse(m.d[3].indicator_var.fixed)
    self.assertFalse(m.d[3].indicator_var.value)
    self.assertTrue(m.d[3].active)
    self.assertIs(m.d[3].ctype, Disjunct)
    self.assertTrue(m.disjunction2.active)