import numpy as np
from ._optimize import OptimizeResult, OptimizeWarning
from warnings import warn
from ._linprog_highs import _linprog_highs
from ._linprog_ip import _linprog_ip
from ._linprog_simplex import _linprog_simplex
from ._linprog_rs import _linprog_rs
from ._linprog_doc import (_linprog_highs_doc, _linprog_ip_doc,  # noqa: F401
from ._linprog_util import (
from copy import deepcopy
def linprog_terse_callback(res):
    """
    A sample callback function demonstrating the linprog callback interface.
    This callback produces brief output to sys.stdout before each iteration
    and after the final iteration of the simplex algorithm.

    Parameters
    ----------
    res : A `scipy.optimize.OptimizeResult` consisting of the following fields:

        x : 1-D array
            The independent variable vector which optimizes the linear
            programming problem.
        fun : float
            Value of the objective function.
        success : bool
            True if the algorithm succeeded in finding an optimal solution.
        slack : 1-D array
            The values of the slack variables. Each slack variable corresponds
            to an inequality constraint. If the slack is zero, then the
            corresponding constraint is active.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints, that is,
            ``b - A_eq @ x``.
        phase : int
            The phase of the optimization being executed. In phase 1 a basic
            feasible solution is sought and the T has an additional row
            representing an alternate objective function.
        status : int
            An integer representing the exit status of the optimization::

                 0 : Optimization terminated successfully
                 1 : Iteration limit reached
                 2 : Problem appears to be infeasible
                 3 : Problem appears to be unbounded
                 4 : Serious numerical difficulties encountered

        nit : int
            The number of iterations performed.
        message : str
            A string descriptor of the exit status of the optimization.
    """
    nit = res['nit']
    x = res['x']
    if nit == 0:
        print('Iter:   X:')
    print(f'{nit: <5d}   ', end='')
    print(x)