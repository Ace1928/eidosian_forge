from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .. import constants
import os
import warnings
def defaultPath(self):
    return self.executableExtension('cplex')