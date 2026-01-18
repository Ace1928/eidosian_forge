import numpy as np
import statsmodels.base.l1_solvers_common as l1_solvers_common
def fit_l1_cvxopt_cp(f, score, start_params, args, kwargs, disp=False, maxiter=100, callback=None, retall=False, full_output=False, hess=None):
    """
    Solve the l1 regularized problem using cvxopt.solvers.cp

    Specifically:  We convert the convex but non-smooth problem

    .. math:: \\min_\\beta f(\\beta) + \\sum_k\\alpha_k |\\beta_k|

    via the transformation to the smooth, convex, constrained problem in twice
    as many variables (adding the "added variables" :math:`u_k`)

    .. math:: \\min_{\\beta,u} f(\\beta) + \\sum_k\\alpha_k u_k,

    subject to

    .. math:: -u_k \\leq \\beta_k \\leq u_k.

    Parameters
    ----------
    All the usual parameters from LikelhoodModel.fit
    alpha : non-negative scalar or numpy array (same size as parameters)
        The weight multiplying the l1 penalty term
    trim_mode : 'auto, 'size', or 'off'
        If not 'off', trim (set to zero) parameters that would have been zero
            if the solver reached the theoretical minimum.
        If 'auto', trim params using the Theory above.
        If 'size', trim params if they have very small absolute value
    size_trim_tol : float or 'auto' (default = 'auto')
        For use when trim_mode === 'size'
    auto_trim_tol : float
        For sue when trim_mode == 'auto'.  Use
    qc_tol : float
        Print warning and do not allow auto trim when (ii) in "Theory" (above)
        is violated by this much.
    qc_verbose : bool
        If true, print out a full QC report upon failure
    abstol : float
        absolute accuracy (default: 1e-7).
    reltol : float
        relative accuracy (default: 1e-6).
    feastol : float
        tolerance for feasibility conditions (default: 1e-7).
    refinement : int
        number of iterative refinement steps when solving KKT equations
        (default: 1).
    """
    from cvxopt import solvers, matrix
    start_params = np.array(start_params).ravel('F')
    k_params = len(start_params)
    x0 = np.append(start_params, np.fabs(start_params))
    x0 = matrix(x0, (2 * k_params, 1))
    alpha = np.array(kwargs['alpha_rescaled']).ravel('F')
    alpha = alpha * np.ones(k_params)
    assert alpha.min() >= 0
    f_0 = lambda x: _objective_func(f, x, k_params, alpha, *args)
    Df = lambda x: _fprime(score, x, k_params, alpha)
    G = _get_G(k_params)
    h = matrix(0.0, (2 * k_params, 1))
    H = lambda x, z: _hessian_wrapper(hess, x, z, k_params)

    def F(x=None, z=None):
        if x is None:
            return (0, x0)
        elif z is None:
            return (f_0(x), Df(x))
        else:
            return (f_0(x), Df(x), H(x, z))
    solvers.options['show_progress'] = disp
    solvers.options['maxiters'] = maxiter
    if 'abstol' in kwargs:
        solvers.options['abstol'] = kwargs['abstol']
    if 'reltol' in kwargs:
        solvers.options['reltol'] = kwargs['reltol']
    if 'feastol' in kwargs:
        solvers.options['feastol'] = kwargs['feastol']
    if 'refinement' in kwargs:
        solvers.options['refinement'] = kwargs['refinement']
    results = solvers.cp(F, G, h)
    x = np.asarray(results['x']).ravel()
    params = x[:k_params]
    qc_tol = kwargs['qc_tol']
    qc_verbose = kwargs['qc_verbose']
    passed = l1_solvers_common.qc_results(params, alpha, score, qc_tol, qc_verbose)
    trim_mode = kwargs['trim_mode']
    size_trim_tol = kwargs['size_trim_tol']
    auto_trim_tol = kwargs['auto_trim_tol']
    params, trimmed = l1_solvers_common.do_trim_params(params, k_params, alpha, score, passed, trim_mode, size_trim_tol, auto_trim_tol)
    if full_output:
        fopt = f_0(x)
        gopt = float('nan')
        hopt = float('nan')
        iterations = float('nan')
        converged = results['status'] == 'optimal'
        warnflag = results['status']
        retvals = {'fopt': fopt, 'converged': converged, 'iterations': iterations, 'gopt': gopt, 'hopt': hopt, 'trimmed': trimmed, 'warnflag': warnflag}
    else:
        x = np.array(results['x']).ravel()
        params = x[:k_params]
    if full_output:
        return (params, retvals)
    else:
        return params