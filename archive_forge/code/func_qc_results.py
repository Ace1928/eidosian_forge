import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def qc_results(params, alpha, score, qc_tol, qc_verbose=False):
    """
    Theory dictates that one of two conditions holds:
        i) abs(score[i]) == alpha[i]  and  params[i] != 0
        ii) abs(score[i]) <= alpha[i]  and  params[i] == 0
    qc_results checks to see that (ii) holds, within qc_tol

    qc_results also checks for nan or results of the wrong shape.

    Parameters
    ----------
    params : ndarray
        model parameters.  Not including the added variables x_added.
    alpha : ndarray
        regularization coefficients
    score : function
        Gradient of unregularized objective function
    qc_tol : float
        Tolerance to hold conditions (i) and (ii) to for QC check.
    qc_verbose : bool
        If true, print out a full QC report upon failure

    Returns
    -------
    passed : bool
        True if QC check passed
    qc_dict : Dictionary
        Keys are fprime, alpha, params, passed_array

    Prints
    ------
    Warning message if QC check fails.
    """
    assert not np.isnan(params).max()
    assert (params == params.ravel('F')).min(), 'params should have already been 1-d'
    fprime = score(params)
    k_params = len(params)
    passed_array = np.array([True] * k_params)
    for i in range(k_params):
        if alpha[i] > 0:
            if (abs(fprime[i]) - alpha[i]) / alpha[i] > qc_tol:
                passed_array[i] = False
    qc_dict = dict(fprime=fprime, alpha=alpha, params=params, passed_array=passed_array)
    passed = passed_array.min()
    if not passed:
        num_failed = (~passed_array).sum()
        message = 'QC check did not pass for %d out of %d parameters' % (num_failed, k_params)
        message += '\nTry increasing solver accuracy or number of iterations, decreasing alpha, or switch solvers'
        if qc_verbose:
            message += _get_verbose_addon(qc_dict)
        import warnings
        warnings.warn(message, ConvergenceWarning)
    return passed