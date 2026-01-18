from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .. import constants
import os
import warnings
def cplex_var_ub(var):
    if var.upBound is not None:
        return float(var.upBound)
    else:
        return cplex.infinity