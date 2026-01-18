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
@property
def n_edges(self):
    """The number of edges in the incidence graph, or the number of
        structural nonzeros in the incidence matrix
        """
    if self._incidence_graph is None:
        raise RuntimeError('Cannot get number of edges (nonzeros) when nothing is cached')
    return len(self._incidence_graph.edges)