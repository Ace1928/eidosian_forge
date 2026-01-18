from collections.abc import Iterable
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect, gurobipy
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.numvalue import value, is_fixed
from pyomo.opt.base import SolverFactory
def get_sos_attr(self, con, attr):
    """
        Get the value of an attribute on a gurobi sos constraint.

        Parameters
        ----------
        con: pyomo.core.base.sos._SOSConstraintData
            The pyomo SOS constraint for which the corresponding gurobi SOS constraint attribute
            should be retrieved.
        attr: str
            The attribute to get. Options are:

                IISSOS
        """
    if self._needs_updated:
        self._update()
    return self._pyomo_con_to_solver_con_map[con].getAttr(attr)