from __future__ import annotations
import warnings
from collections import defaultdict
import numpy as np
import scipy as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cone2cone.affine2direct import Dualize, Slacks
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.utilities import expcone_permutor
@staticmethod
def recover_dual_variables(task, sol):
    if task.getnumintvar() == 0:
        dual_vars = dict()
        dual_var = [0.0] * task.getnumcon()
        task.gety(sol, dual_var)
        dual_vars[s.EQ_DUAL] = np.array(dual_var)
    else:
        dual_vars = None
    return dual_vars