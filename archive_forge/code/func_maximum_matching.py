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
def maximum_matching(self, variables=None, constraints=None):
    """Return a maximum cardinality matching of variables and constraints.

        The matching maps constraints to their matched variables.

        Returns
        -------
        ``ComponentMap``
            A map from constraints to their matched variables.

        """
    variables, constraints = self._validate_input(variables, constraints)
    graph = self._extract_subgraph(variables, constraints)
    con_nodes = list(range(len(constraints)))
    matching = maximum_matching(graph, top_nodes=con_nodes)
    M = len(constraints)
    return ComponentMap(((constraints[i], variables[j - M]) for i, j in matching.items()))