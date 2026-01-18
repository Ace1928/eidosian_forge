import pyomo.common.unittest as unittest
from pyomo.contrib.gdp_bounds.info import disjunctive_lb, disjunctive_ub
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import check_available_solvers
def test_nested_fbbt(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 8))
    m.y = Var(bounds=(0, 8))
    m.d1 = Disjunct()
    m.d1.c = Constraint(expr=m.x >= 2)
    m.d1.innerD1 = Disjunct()
    m.d1.innerD1.c = Constraint(expr=m.y >= m.x + 3)
    m.d1.innerD2 = Disjunct()
    m.d1.innerD2.c = Constraint(expr=m.y <= m.x - 4)
    m.d1.innerDisj = Disjunction(expr=[m.d1.innerD1, m.d1.innerD2])
    m.d2 = Disjunct()
    m.d2.c = Constraint(expr=m.x == 3)
    m.disj = Disjunction(expr=[m.d1, m.d2])
    TransformationFactory('contrib.compute_disj_var_bounds').apply_to(m)
    self.assertEqual(disjunctive_lb(m.y, m.d1), 0)
    self.assertEqual(disjunctive_lb(m.y, m.d1.innerD1), 5)
    self.assertEqual(disjunctive_ub(m.y, m.d1.innerD2), 4)