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
def test_bipartite_incidence_graph(self):
    m = self.make_test_model()
    constraints = [m.eq1, m.eq2, m.ineq1, m.ineq2, m.ineq3]
    variables = list(m.v.values())
    graph = get_bipartite_incidence_graph(variables, constraints)
    self.assertEqual(len(graph.nodes), 9)
    self.assertEqual(len(graph.edges), 11)
    self.assertTrue(nx.algorithms.bipartite.is_bipartite(graph))
    self.assertEqual(set(graph[0]), {5, 6})
    self.assertEqual(set(graph[1]), {5, 7})
    self.assertEqual(set(graph[2]), {6, 7, 8})
    self.assertEqual(set(graph[3]), {6, 8})
    self.assertEqual(set(graph[4]), {5, 8})
    self.assertEqual(set(graph[5]), {0, 1, 4})
    self.assertEqual(set(graph[6]), {0, 2, 3})
    self.assertEqual(set(graph[7]), {1, 2})
    self.assertEqual(set(graph[8]), {2, 3, 4})