import pyomo.common.unittest as unittest
from pyomo.contrib.gdp_bounds.info import disjunctive_lb, disjunctive_ub
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import check_available_solvers
def test_compute_bounds_fbbt(self):
    """Test computation of disjunctive bounds."""
    m = ConcreteModel()
    m.x = Var(bounds=(0, 8))
    m.d1 = Disjunct()
    m.d1.c = Constraint(expr=m.x >= 2)
    m.d2 = Disjunct()
    m.d2.c = Constraint(expr=m.x <= 4)
    m.disj = Disjunction(expr=[m.d1, m.d2])
    m.obj = Objective(expr=m.x)
    TransformationFactory('contrib.compute_disj_var_bounds').apply_to(m)
    self.assertEqual(m.d1._disj_var_bounds[m.x], (2, 8))
    self.assertEqual(m.d2._disj_var_bounds[m.x], (0, 4))
    self.assertEqual(disjunctive_lb(m.x, m.d1), 2)
    self.assertEqual(disjunctive_ub(m.x, m.d1), 8)
    self.assertEqual(disjunctive_lb(m.x, m.d2), 0)
    self.assertEqual(disjunctive_ub(m.x, m.d2), 4)