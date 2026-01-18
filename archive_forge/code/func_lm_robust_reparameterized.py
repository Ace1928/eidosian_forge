import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
def lm_robust_reparameterized(score, params_deriv, score_deriv, cov_score):
    """robust generalized score test for transformed parameters

    The parameters are given by a nonlinear transformation of the estimated
    reduced parameters

    `params = g(params_reduced)`  with jacobian `G = d g / d params_reduced`

    score and other arrays are for full parameter space `params`

    Parameters
    ----------
    score : ndarray, 1-D
        derivative of objective function at estimated parameters
        of constrained model
    params_deriv : ndarray
        Jacobian G of the parameter trasnformation
    score_deriv : ndarray, symmetric, square
        second derivative of objective function
        TODO: could be OPG or any other estimator if information matrix
        equality holds
    cov_score B :  ndarray, symmetric, square
        covariance matrix of the score. This is the inner part of a sandwich
        estimator.

    Returns
    -------
    lm_stat : float
        score/lagrange multiplier statistic
    p-value : float
        p-value of the LM test based on chisquare distribution

    Notes
    -----
    Boos 1992, section 4.3, expression for T_{GS} just before example 6
    """
    k_params, k_reduced = params_deriv.shape
    k_constraints = k_params - k_reduced
    G = params_deriv
    tmp_c0 = np.linalg.pinv(G.T.dot(score_deriv.dot(G)))
    tmp_c1 = score_deriv.dot(G.dot(tmp_c0.dot(G.T)))
    tmp_c = np.eye(k_params) - tmp_c1
    cov = tmp_c.dot(cov_score.dot(tmp_c.T))
    lm_stat = score.dot(np.linalg.pinv(cov).dot(score))
    pval = stats.chi2.sf(lm_stat, k_constraints)
    return (lm_stat, pval)