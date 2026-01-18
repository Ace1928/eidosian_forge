from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .core import cbc_path, pulp_cbc_path, coinMP_path, devnull, operating_system
import os
from .. import constants
from tempfile import mktemp
import ctypes
import warnings
def getSolverVersion(self):
    """
            returns a solver version string

            example:
            >>> COINMP_DLL().getSolverVersion() # doctest: +ELLIPSIS
            '...'
            """
    return self.lib.CoinGetVersionStr()