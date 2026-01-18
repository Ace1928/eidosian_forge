import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
from cvxpy.utilities.versioning import Version
@staticmethod
def parse_solver_options(solver_opts):
    import scs
    if Version(scs.__version__) < Version('3.0.0'):
        if 'eps_abs' in solver_opts or 'eps_rel' in solver_opts:
            solver_opts['eps'] = min(solver_opts.get('eps_abs', 1), solver_opts.get('eps_rel', 1))
        else:
            solver_opts['eps'] = solver_opts.get('eps', 0.0001)
    elif 'eps' in solver_opts:
        solver_opts['eps_abs'] = solver_opts['eps']
        solver_opts['eps_rel'] = solver_opts['eps']
        del solver_opts['eps']
    else:
        solver_opts['eps_abs'] = solver_opts.get('eps_abs', 1e-05)
        solver_opts['eps_rel'] = solver_opts.get('eps_rel', 1e-05)
    if 'use_quad_obj' in solver_opts:
        del solver_opts['use_quad_obj']
    return solver_opts