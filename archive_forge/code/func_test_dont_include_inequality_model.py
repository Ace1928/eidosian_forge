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
def test_dont_include_inequality_model(self):
    m = self.make_model_with_inequalities()
    igraph = IncidenceGraphInterface(m, include_inequality=False)
    self.assertEqual(igraph.incidence_matrix.shape, (8, 8))