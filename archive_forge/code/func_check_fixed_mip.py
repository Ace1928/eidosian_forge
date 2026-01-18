from pyomo.common.errors import InfeasibleConstraintException
import pyomo.common.unittest as unittest
from pyomo.environ import Block, ConcreteModel, Constraint, TransformationFactory
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error
def check_fixed_mip(self, m):
    self.assertTrue(m.d1.indicator_var.fixed)
    self.assertTrue(m.d1.active)
    self.assertIs(m.d1.ctype, Block)
    self.assertTrue(m.d2.indicator_var.fixed)
    self.assertFalse(m.d2.active)
    self.assertFalse(m.disjunction1.active)