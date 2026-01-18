import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.dependencies import (
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import (
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import (
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def test_assumed_constraint_behavior(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3])
    m.con = pyo.Constraint(expr=m.x[1] == m.x[2] - pyo.exp(m.x[3]))
    var_set = ComponentSet(identify_variables(m.con.body))
    self.assertEqual(var_set, ComponentSet(m.x[:]))