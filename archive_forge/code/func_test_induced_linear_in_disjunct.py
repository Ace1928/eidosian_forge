import pyomo.common.unittest as unittest
from pyomo.contrib.preprocessing.plugins.induced_linearity import (
from pyomo.common.collections import ComponentSet, Bunch
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn
@unittest.skipIf(not glpk_available, 'GLPK not available')
def test_induced_linear_in_disjunct(self):
    m = ConcreteModel()
    m.x = Var([0], bounds=(-3, 8))
    m.y = Var(RangeSet(2), domain=Binary)
    m.logical = ConstraintList()
    m.logical.add(expr=m.y[1] + m.y[2] == 1)
    m.v = Var([1])
    m.v[1].setlb(-2)
    m.v[1].setub(7)
    m.bilinear_outside = Constraint(expr=m.x[0] * m.v[1] >= 2)
    m.disjctn = Disjunction(expr=[[m.x[0] * m.v[1] == 3, 2 * m.x[0] == m.y[1] + m.y[2]], [m.x[0] * m.v[1] == 4]])
    TransformationFactory('contrib.induced_linearity').apply_to(m)
    self.assertEqual(m.disjctn.disjuncts[0].constraint[1].body.polynomial_degree(), 1)
    self.assertEqual(m.bilinear_outside.body.polynomial_degree(), 2)
    self.assertEqual(m.disjctn.disjuncts[1].constraint[1].body.polynomial_degree(), 2)