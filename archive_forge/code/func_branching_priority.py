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
def branching_priority(self):
    """
        Advanced usage: Certain MIP solvers (e.g. Gurobi or SCIP) allow you to set
        a per-variable priority for determining which variable to branch on.

        A value of 0 is treated as default, and is equivalent to not setting the
        branching priority. The solver looks first to branch on fractional
        variables in higher priority levels. As of 2019-05, only Gurobi and SCIP
        support setting branching priority; all other solvers will simply ignore
        this annotation.
        """
    return _pywraplp.Variable_branching_priority(self)