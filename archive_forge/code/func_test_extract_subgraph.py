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
def test_extract_subgraph(self):
    m = self.make_test_model()
    constraints = [m.eq1, m.eq2, m.ineq1, m.ineq2, m.ineq3]
    variables = list(m.v.values())
    graph = get_bipartite_incidence_graph(variables, constraints)
    sg_cons = [0, 2]
    sg_vars = [i + len(constraints) for i in [2, 0, 3]]
    subgraph = extract_bipartite_subgraph(graph, sg_cons, sg_vars)
    self.assertEqual(len(subgraph.nodes), 5)
    self.assertEqual(len(subgraph.edges), 3)
    self.assertTrue(nx.algorithms.bipartite.is_bipartite(subgraph))
    self.assertEqual(set(subgraph[0]), {3})
    self.assertEqual(set(subgraph[1]), {2, 4})
    self.assertEqual(set(subgraph[2]), {1})
    self.assertEqual(set(subgraph[3]), {0})
    self.assertEqual(set(subgraph[4]), {1})