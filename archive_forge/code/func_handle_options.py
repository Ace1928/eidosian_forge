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
def handle_options(env, task, verbose: bool, solver_opts: dict) -> dict:
    """
        Handle user-specified solver options.

        Options that have to be applied before the optimization are applied to the task here.
        A new dictionary is returned with the processed options and default options applied.
        """
    import mosek
    if verbose:

        def streamprinter(text):
            s.LOGGER.info(text.rstrip('\n'))
        print('\n')
        env.set_Stream(mosek.streamtype.log, streamprinter)
        task.set_Stream(mosek.streamtype.log, streamprinter)
    solver_opts = MOSEK.parse_eps_keyword(solver_opts)
    mosek_params = solver_opts.pop('mosek_params', dict())
    if any((MOSEK.is_param(p) for p in mosek_params)):
        warnings.warn(__MSK_ENUM_PARAM_DEPRECATION__, DeprecationWarning)
        warnings.warn(__MSK_ENUM_PARAM_DEPRECATION__, UserWarning)
    for param, value in mosek_params.items():
        if isinstance(param, str):
            param = param.strip()
            if isinstance(value, str):
                task.putparam(param, value)
            elif param.startswith('MSK_DPAR_'):
                task.putnadouparam(param, value)
            elif param.startswith('MSK_IPAR_'):
                task.putnaintparam(param, value)
            elif param.startswith('MSK_SPAR_'):
                task.putnastrparam(param, value)
            else:
                raise ValueError("Invalid MOSEK parameter '%s'." % param)
        elif isinstance(param, mosek.dparam):
            task.putdouparam(param, value)
        elif isinstance(param, mosek.iparam):
            task.putintparam(param, value)
        elif isinstance(param, mosek.sparam):
            task.putstrparam(param, value)
        else:
            raise ValueError("Invalid MOSEK parameter '%s'." % param)
    processed_opts = dict()
    processed_opts['mosek_params'] = mosek_params
    processed_opts['save_file'] = solver_opts.pop('save_file', False)
    processed_opts['bfs'] = solver_opts.pop('bfs', False)
    processed_opts['accept_unknown'] = solver_opts.pop('accept_unknown', False)
    if solver_opts:
        raise ValueError(f'Invalid keyword-argument(s) {solver_opts.keys()} passed to MOSEK solver.')
    if processed_opts['bfs']:
        task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.always)
    else:
        task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
    return processed_opts