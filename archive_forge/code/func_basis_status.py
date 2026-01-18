from sys import version_info as _swig_python_version_info
import numbers
from ortools.linear_solver.python.linear_solver_natural_api import OFFSET_KEY
from ortools.linear_solver.python.linear_solver_natural_api import inf
from ortools.linear_solver.python.linear_solver_natural_api import LinearExpr
from ortools.linear_solver.python.linear_solver_natural_api import ProductCst
from ortools.linear_solver.python.linear_solver_natural_api import Sum
from ortools.linear_solver.python.linear_solver_natural_api import SumArray
from ortools.linear_solver.python.linear_solver_natural_api import SumCst
from ortools.linear_solver.python.linear_solver_natural_api import LinearConstraint
from ortools.linear_solver.python.linear_solver_natural_api import VariableExpr
def basis_status(self):
    """
        Advanced usage: returns the basis status of the constraint.

        It is only available for continuous problems).

        Note that if a constraint "linear_expression in [lb, ub]" is transformed
        into "linear_expression + slack = 0" with slack in [-ub, -lb], then this
        status is the same as the status of the slack variable with AT_UPPER_BOUND
        and AT_LOWER_BOUND swapped.

        See also: MPSolver::BasisStatus.
        """
    return _pywraplp.Constraint_basis_status(self)