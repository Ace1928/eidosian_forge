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
def test_diagonal_blocks_with_cached_maps(self):
    N = 5
    model = make_gas_expansion_model(N)
    igraph = IncidenceGraphInterface()
    variables = []
    variables.extend(model.P.values())
    variables.extend((model.T[i] for i in model.streams if i != model.streams.first()))
    variables.extend((model.rho[i] for i in model.streams if i != model.streams.first()))
    variables.extend((model.F[i] for i in model.streams if i != model.streams.first()))
    constraints = list(model.component_data_objects(pyo.Constraint))
    igraph.block_triangularize(variables, constraints)
    var_blocks, con_blocks = igraph.get_diagonal_blocks(variables, constraints)
    self.assertIs(igraph.row_block_map, None)
    self.assertIs(igraph.col_block_map, None)