import pyomo.common.unittest as unittest
from pyomo.contrib.preprocessing.plugins.induced_linearity import (
from pyomo.common.collections import ComponentSet, Bunch
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn
def test_detect_effectively_discrete_vars(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var(domain=Binary)
    m.z = Var(domain=Integers)
    m.constr = Constraint(expr=m.x == m.y + m.z)
    m.ignore_inequality = Constraint(expr=m.x <= m.y + m.z)
    m.ignore_nonlinear = Constraint(expr=m.x ** 2 == m.y + m.z)
    m.a = Var()
    m.b = Var(domain=Binary)
    m.c = Var(domain=Integers)
    m.disj = Disjunct()
    m.disj.constr = Constraint(expr=m.a == m.b + m.c)
    effectively_discrete = detect_effectively_discrete_vars(m, 1e-06)
    self.assertEqual(len(effectively_discrete), 1)
    self.assertEqual(effectively_discrete[m.x], [m.constr])
    effectively_discrete = detect_effectively_discrete_vars(m.disj, 1e-06)
    self.assertEqual(len(effectively_discrete), 1)
    self.assertEqual(effectively_discrete[m.a], [m.disj.constr])