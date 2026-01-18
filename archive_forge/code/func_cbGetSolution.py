from collections.abc import Iterable
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect, gurobipy
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.numvalue import value, is_fixed
from pyomo.opt.base import SolverFactory
def cbGetSolution(self, vars):
    """
        Parameters
        ----------
        vars: iterable of vars
        """
    StaleFlagManager.mark_all_as_stale()
    if not isinstance(vars, Iterable):
        vars = [vars]
    gurobi_vars = [self._pyomo_var_to_solver_var_map[i] for i in vars]
    var_values = self._solver_model.cbGetSolution(gurobi_vars)
    for i, v in enumerate(vars):
        v.set_value(var_values[i], skip_validation=True)