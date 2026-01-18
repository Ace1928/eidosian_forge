import enum
import textwrap
from pyomo.core.base.block import _BlockData
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.expr import EqualityExpression
from pyomo.util.subsystems import create_subsystem_block
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.dependencies import (
from pyomo.common.deprecation import deprecated
from pyomo.contrib.incidence_analysis.config import get_config_from_kwds
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.connected import get_independent_submatrices
from pyomo.contrib.incidence_analysis.triangularize import (
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import (
from pyomo.contrib.incidence_analysis.incidence import get_incident_variables
from pyomo.contrib.pynumero.asl import AmplInterface
def map_nodes_to_block_triangular_indices(self, variables=None, constraints=None):
    """Map variables and constraints to indices of their diagonal blocks in
        a block lower triangular permutation

        Returns
        -------
        var_block_map: ``ComponentMap``
            Map from variables to their diagonal blocks in a block
            triangularization
        con_block_map: ``ComponentMap``
            Map from constraints to their diagonal blocks in a block
            triangularization

        """
    variables, constraints = self._validate_input(variables, constraints)
    graph = self._extract_subgraph(variables, constraints)
    M = len(constraints)
    con_nodes = list(range(M))
    sccs = get_scc_of_projection(graph, con_nodes)
    row_idx_map = {r: idx for idx, scc in enumerate(sccs) for r, _ in scc}
    col_idx_map = {c - M: idx for idx, scc in enumerate(sccs) for _, c in scc}
    con_block_map = ComponentMap(((constraints[i], idx) for i, idx in row_idx_map.items()))
    var_block_map = ComponentMap(((variables[j], idx) for j, idx in col_idx_map.items()))
    return (var_block_map, con_block_map)