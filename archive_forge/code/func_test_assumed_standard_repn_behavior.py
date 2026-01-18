import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.incidence import (
def test_assumed_standard_repn_behavior(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2])
    m.p = pyo.Param(initialize=0.0)
    expr = m.x[1] + 0 * m.x[2]
    repn = generate_standard_repn(expr)
    self.assertEqual(len(repn.linear_vars), 1)
    self.assertIs(repn.linear_vars[0], m.x[1])
    expr = m.p * m.x[1] + m.x[2]
    repn = generate_standard_repn(expr)
    self.assertEqual(len(repn.linear_vars), 1)
    self.assertIs(repn.linear_vars[0], m.x[2])