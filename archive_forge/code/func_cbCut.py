from collections.abc import Iterable
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect, gurobipy
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.numvalue import value, is_fixed
from pyomo.opt.base import SolverFactory
def cbCut(self, con):
    """
        Add a cut within a callback.

        Parameters
        ----------
        con: pyomo.core.base.constraint._GeneralConstraintData
            The cut to add
        """
    if not con.active:
        raise ValueError('cbCut expected an active constraint.')
    if is_fixed(con.body):
        raise ValueError('cbCut expected a non-trivial constraint')
    gurobi_expr, referenced_vars = self._get_expr_from_pyomo_expr(con.body, self._max_constraint_degree)
    if con.has_lb():
        if con.has_ub():
            raise ValueError('Range constraints are not supported in cbCut.')
        if not is_fixed(con.lower):
            raise ValueError('Lower bound of constraint {0} is not constant.'.format(con))
    if con.has_ub():
        if not is_fixed(con.upper):
            raise ValueError('Upper bound of constraint {0} is not constant.'.format(con))
    if con.equality:
        self._solver_model.cbCut(lhs=gurobi_expr, sense=gurobipy.GRB.EQUAL, rhs=value(con.lower))
    elif con.has_lb() and value(con.lower) > -float('inf'):
        self._solver_model.cbCut(lhs=gurobi_expr, sense=gurobipy.GRB.GREATER_EQUAL, rhs=value(con.lower))
    elif con.has_ub() and value(con.upper) < float('inf'):
        self._solver_model.cbCut(lhs=gurobi_expr, sense=gurobipy.GRB.LESS_EQUAL, rhs=value(con.upper))
    else:
        raise ValueError('Constraint does not have a lower or an upper bound {0} \n'.format(con))