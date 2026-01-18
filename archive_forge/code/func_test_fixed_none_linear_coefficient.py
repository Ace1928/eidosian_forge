import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.incidence import (
def test_fixed_none_linear_coefficient(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3])
    m.p = pyo.Param([1, 2], mutable=True, initialize=1.0)
    m.x[3].fix(None)
    expr = 2 * m.x[1] + 3 * m.x[3] * m.p[2] * m.x[2] + m.x[1] ** 2
    variables = self._get_incident_variables(expr)
    self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1], m.x[2]]))