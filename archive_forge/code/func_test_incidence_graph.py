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
@unittest.skipUnless(scipy_available, 'scipy is not available.')
def test_incidence_graph(self):
    m = make_degenerate_solid_phase_model()
    variables = list(m.component_data_objects(pyo.Var))
    constraints = list(m.component_data_objects(pyo.Constraint))
    graph = get_incidence_graph(variables, constraints)
    matrix = get_structural_incidence_matrix(variables, constraints)
    from_matrix = from_biadjacency_matrix(matrix)
    self.assertEqual(graph.nodes, from_matrix.nodes)
    self.assertEqual(graph.edges, from_matrix.edges)