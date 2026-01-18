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
def test_imperfect_matching(self):
    model = make_gas_expansion_model()
    igraph = IncidenceGraphInterface()
    constraints = list(model.component_data_objects(pyo.Constraint))
    variables = list(model.component_data_objects(pyo.Var))
    n_eqn = len(constraints)
    matching = igraph.maximum_matching(variables, constraints)
    values = ComponentSet(matching.values())
    self.assertEqual(len(matching), n_eqn)
    self.assertEqual(len(values), n_eqn)