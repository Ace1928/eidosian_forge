from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .. import constants
import os
import warnings
def getOptions(self):
    params_eq = dict(logPath='set logFile {}', gapRel='set mip tolerances mipgap {}', gapAbs='set mip tolerances absmipgap {}', maxMemory='set mip limits treememory {}', threads='set threads {}', maxNodes='set mip limits nodes {}')
    return [v.format(self.optionsDict[k]) for k, v in params_eq.items() if k in self.optionsDict and self.optionsDict[k] is not None]