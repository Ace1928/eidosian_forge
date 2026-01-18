import pyomo.common.unittest as unittest
from pyomo.contrib.preprocessing.plugins.induced_linearity import (
from pyomo.common.collections import ComponentSet, Bunch
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn
def test_detect_bilinear_vars(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.z = Var()
    m.c = Constraint(expr=(m.x - 3) * (m.y + 2) - (m.z + 4) * m.y + (m.x + 2) ** 2 + exp(m.y ** 2) * m.x <= m.z)
    m.c2 = Constraint(expr=m.x * m.y == 3)
    bilinear_map = _bilinear_expressions(m)
    self.assertEqual(len(bilinear_map), 3)
    self.assertEqual(len(bilinear_map[m.x]), 2)
    self.assertEqual(len(bilinear_map[m.y]), 2)
    self.assertEqual(len(bilinear_map[m.z]), 1)
    self.assertEqual(bilinear_map[m.x][m.x], ComponentSet([m.c]))
    self.assertEqual(bilinear_map[m.x][m.y], ComponentSet([m.c, m.c2]))
    self.assertEqual(bilinear_map[m.y][m.x], ComponentSet([m.c, m.c2]))
    self.assertEqual(bilinear_map[m.y][m.z], ComponentSet([m.c]))
    self.assertEqual(bilinear_map[m.z][m.y], ComponentSet([m.c]))