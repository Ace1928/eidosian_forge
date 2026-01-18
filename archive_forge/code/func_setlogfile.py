from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .. import constants
import os
import warnings
def setlogfile(self, fileobj):
    """
            sets the logfile for cplex output
            """
    self.solverModel.set_error_stream(fileobj)
    self.solverModel.set_log_stream(fileobj)
    self.solverModel.set_warning_stream(fileobj)
    self.solverModel.set_results_stream(fileobj)