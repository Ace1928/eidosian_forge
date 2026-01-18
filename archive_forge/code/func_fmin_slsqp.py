import numpy as np
from scipy.optimize._slsqp import slsqp
from numpy import (zeros, array, linalg, append, concatenate, finfo,
from ._optimize import (OptimizeResult, _check_unknown_options,
from ._numdiff import approx_derivative
from ._constraints import old_bound_to_new, _arr_to_scalar
from scipy._lib._array_api import atleast_nd, array_namespace
from numpy import exp, inf  # noqa: F401
def fmin_slsqp(func, x0, eqcons=(), f_eqcons=None, ieqcons=(), f_ieqcons=None, bounds=(), fprime=None, fprime_eqcons=None, fprime_ieqcons=None, args=(), iter=100, acc=1e-06, iprint=1, disp=None, full_output=0, epsilon=_epsilon, callback=None):
    """
    Minimize a function using Sequential Least Squares Programming

    Python interface function for the SLSQP Optimization subroutine
    originally implemented by Dieter Kraft.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function.  Must return a scalar.
    x0 : 1-D ndarray of float
        Initial guess for the independent variable(s).
    eqcons : list, optional
        A list of functions of length n such that
        eqcons[j](x,*args) == 0.0 in a successfully optimized
        problem.
    f_eqcons : callable f(x,*args), optional
        Returns a 1-D array in which each element must equal 0.0 in a
        successfully optimized problem. If f_eqcons is specified,
        eqcons is ignored.
    ieqcons : list, optional
        A list of functions of length n such that
        ieqcons[j](x,*args) >= 0.0 in a successfully optimized
        problem.
    f_ieqcons : callable f(x,*args), optional
        Returns a 1-D ndarray in which each element must be greater or
        equal to 0.0 in a successfully optimized problem. If
        f_ieqcons is specified, ieqcons is ignored.
    bounds : list, optional
        A list of tuples specifying the lower and upper bound
        for each independent variable [(xl0, xu0),(xl1, xu1),...]
        Infinite values will be interpreted as large floating values.
    fprime : callable `f(x,*args)`, optional
        A function that evaluates the partial derivatives of func.
    fprime_eqcons : callable `f(x,*args)`, optional
        A function of the form `f(x, *args)` that returns the m by n
        array of equality constraint normals. If not provided,
        the normals will be approximated. The array returned by
        fprime_eqcons should be sized as ( len(eqcons), len(x0) ).
    fprime_ieqcons : callable `f(x,*args)`, optional
        A function of the form `f(x, *args)` that returns the m by n
        array of inequality constraint normals. If not provided,
        the normals will be approximated. The array returned by
        fprime_ieqcons should be sized as ( len(ieqcons), len(x0) ).
    args : sequence, optional
        Additional arguments passed to func and fprime.
    iter : int, optional
        The maximum number of iterations.
    acc : float, optional
        Requested accuracy.
    iprint : int, optional
        The verbosity of fmin_slsqp :

        * iprint <= 0 : Silent operation
        * iprint == 1 : Print summary upon completion (default)
        * iprint >= 2 : Print status of each iterate and summary
    disp : int, optional
        Overrides the iprint interface (preferred).
    full_output : bool, optional
        If False, return only the minimizer of func (default).
        Otherwise, output final objective function and summary
        information.
    epsilon : float, optional
        The step size for finite-difference derivative estimates.
    callback : callable, optional
        Called after each iteration, as ``callback(x)``, where ``x`` is the
        current parameter vector.

    Returns
    -------
    out : ndarray of float
        The final minimizer of func.
    fx : ndarray of float, if full_output is true
        The final value of the objective function.
    its : int, if full_output is true
        The number of iterations.
    imode : int, if full_output is true
        The exit mode from the optimizer (see below).
    smode : string, if full_output is true
        Message describing the exit mode from the optimizer.

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'SLSQP' `method` in particular.

    Notes
    -----
    Exit modes are defined as follows ::

        -1 : Gradient evaluation required (g & a)
         0 : Optimization terminated successfully
         1 : Function evaluation required (f & c)
         2 : More equality constraints than independent variables
         3 : More than 3*n iterations in LSQ subproblem
         4 : Inequality constraints incompatible
         5 : Singular matrix E in LSQ subproblem
         6 : Singular matrix C in LSQ subproblem
         7 : Rank-deficient equality constraint subproblem HFTI
         8 : Positive directional derivative for linesearch
         9 : Iteration limit reached

    Examples
    --------
    Examples are given :ref:`in the tutorial <tutorial-sqlsp>`.

    """
    if disp is not None:
        iprint = disp
    opts = {'maxiter': iter, 'ftol': acc, 'iprint': iprint, 'disp': iprint != 0, 'eps': epsilon, 'callback': callback}
    cons = ()
    cons += tuple(({'type': 'eq', 'fun': c, 'args': args} for c in eqcons))
    cons += tuple(({'type': 'ineq', 'fun': c, 'args': args} for c in ieqcons))
    if f_eqcons:
        cons += ({'type': 'eq', 'fun': f_eqcons, 'jac': fprime_eqcons, 'args': args},)
    if f_ieqcons:
        cons += ({'type': 'ineq', 'fun': f_ieqcons, 'jac': fprime_ieqcons, 'args': args},)
    res = _minimize_slsqp(func, x0, args, jac=fprime, bounds=bounds, constraints=cons, **opts)
    if full_output:
        return (res['x'], res['fun'], res['nit'], res['status'], res['message'])
    else:
        return res['x']