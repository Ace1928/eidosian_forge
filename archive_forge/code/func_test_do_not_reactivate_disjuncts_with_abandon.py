import pyomo.common.unittest as unittest
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
def test_do_not_reactivate_disjuncts_with_abandon(self):
    m = ConcreteModel()
    m.x = Var()
    m.s = RangeSet(4)
    m.d = Disjunct(m.s)
    m.d[2].bad_constraint_should_not_be_active = Constraint(expr=m.x >= 1)
    m.disj1 = Disjunction(expr=[m.d[1], m.d[2]])
    m.disj2 = Disjunction(expr=[m.d[3], m.d[4]])
    m.d[1].indicator_var.fix(1)
    m.d[2].deactivate()
    TransformationFactory('gdp.bigm').apply_to(m)
    self.assertFalse(m.d[2].active)