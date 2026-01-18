import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.incidence import (
def test_fixed_zero_linear_coefficient(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3])
    m.p = pyo.Param([1, 2], mutable=True, initialize=1.0)
    m.p[1].set_value(0)
    expr = 2 * m.x[1] + m.p[1] * m.p[2] * m.x[2] + m.p[2] * m.x[3] ** 2
    variables = self._get_incident_variables(expr)
    self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1], m.x[3]]))
    m.x[3].fix(0.0)
    expr = 2 * m.x[1] + 3 * m.x[3] * m.p[2] * m.x[2] + m.x[1] ** 2
    variables = self._get_incident_variables(expr)
    self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1]]))
    m.x[3].fix(1.0)
    variables = self._get_incident_variables(expr)
    self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1], m.x[2]]))