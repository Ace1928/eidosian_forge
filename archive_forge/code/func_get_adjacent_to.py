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
def get_adjacent_to(self, component):
    """Return a list of components adjacent to the provided component
        in the cached bipartite incidence graph of variables and constraints

        Parameters
        ----------
        component: ``ComponentData``
            The variable or constraint data object whose adjacent components
            are returned

        Returns
        -------
        list of ComponentData
            List of constraint or variable data objects adjacent to the
            provided component

        Example
        -------

        .. doctest::
           :skipif: not networkx_available

           >>> import pyomo.environ as pyo
           >>> from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
           >>> m = pyo.ConcreteModel()
           >>> m.x = pyo.Var([1, 2])
           >>> m.eq1 = pyo.Constraint(expr=m.x[1]**2 == 7)
           >>> m.eq2 = pyo.Constraint(expr=m.x[1]*m.x[2] == 3)
           >>> m.eq3 = pyo.Constraint(expr=m.x[1] + 2*m.x[2] == 5)
           >>> igraph = IncidenceGraphInterface(m)
           >>> adj_to_x2 = igraph.get_adjacent_to(m.x[2])
           >>> print([c.name for c in adj_to_x2])
           ['eq2', 'eq3']

        """
    if self._incidence_graph is None:
        raise RuntimeError('Cannot get components adjacent to %s if an incidence graph is not cached.' % component)
    _check_unindexed([component])
    M = len(self.constraints)
    N = len(self.variables)
    if component in self._var_index_map:
        vnode = M + self._var_index_map[component]
        adj = self._incidence_graph[vnode]
        adj_comps = [self.constraints[i] for i in adj]
    elif component in self._con_index_map:
        cnode = self._con_index_map[component]
        adj = self._incidence_graph[cnode]
        adj_comps = [self.variables[j - M] for j in adj]
    else:
        raise RuntimeError('Cannot find component %s in the cached incidence graph.' % component)
    return adj_comps