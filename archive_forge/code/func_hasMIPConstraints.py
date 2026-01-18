from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock
from .core import glpk_path, operating_system, log
import os
from .. import constants
def hasMIPConstraints(self, solverModel):
    return glpk.glp_get_num_int(solverModel) > 0 or glpk.glp_get_num_bin(solverModel) > 0