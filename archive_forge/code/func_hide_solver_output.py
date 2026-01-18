from collections import namedtuple
from operator import attrgetter
import numpy as np
from scipy.sparse import dok_matrix
import cvxpy.settings as s
from cvxpy.constraints import SOC
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
def hide_solver_output(model) -> None:
    """Set CPLEX verbosity level (either on or off)."""
    model.set_results_stream(None)
    model.set_warning_stream(None)
    model.set_error_stream(None)
    model.set_log_stream(None)