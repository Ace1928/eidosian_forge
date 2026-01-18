from math import inf
from typing import List
from .core import LpSolver, LpSolver_CMD, subprocess, PulpSolverError
import os, sys
from .. import constants
def createAndConfigureSolver(self, lp):
    lp.solverModel = highspy.Highs()
    if self.msg or self.callbackTuple:
        callbackTuple = self.callbackTuple or (HiGHS.DEFAULT_CALLBACK, HiGHS.DEFAULT_CALLBACK_VALUE)
        lp.solverModel.setLogCallback(*callbackTuple)
    if self.gapRel is not None:
        lp.solverModel.setOptionValue('mip_rel_gap', self.gapRel)
    if self.gapAbs is not None:
        lp.solverModel.setOptionValue('mip_abs_gap', self.gapAbs)
    if self.threads is not None:
        lp.solverModel.setOptionValue('threads', self.threads)
    if self.timeLimit is not None:
        lp.solverModel.setOptionValue('time_limit', float(self.timeLimit))
    for key, value in self.optionsDict.items():
        lp.solverModel.setOptionValue(key, value)