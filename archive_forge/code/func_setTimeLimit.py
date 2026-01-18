from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .. import constants
import os
import warnings
def setTimeLimit(self, timeLimit=0.0):
    """
            Make cplex limit the time it takes --added CBM 8/28/09
            """
    self.solverModel.parameters.timelimit.set(timeLimit)