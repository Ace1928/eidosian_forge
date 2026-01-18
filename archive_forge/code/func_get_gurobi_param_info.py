from collections.abc import Iterable
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect, gurobipy
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.numvalue import value, is_fixed
from pyomo.opt.base import SolverFactory
def get_gurobi_param_info(self, param):
    """
        Get information about a gurobi parameter.

        Parameters
        ----------
        param: str
            The gurobi parameter to get info for. See Gurobi documentation for possible options.

        Returns
        -------
        six-tuple containing the parameter name, type, value, minimum value, maximum value, and default value.
        """
    return self._solver_model.getParamInfo(param)