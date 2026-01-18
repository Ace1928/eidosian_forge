import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def cov_cluster(results, group, use_correction=True):
    """cluster robust covariance matrix

    Calculates sandwich covariance matrix for a single cluster, i.e. grouped
    variables.

    Parameters
    ----------
    results : result instance
       result of a regression, uses results.model.exog and results.resid
       TODO: this should use wexog instead
    use_correction : bool
       If true (default), then the small sample correction factor is used.

    Returns
    -------
    cov : ndarray, (k_vars, k_vars)
        cluster robust covariance matrix for parameter estimates

    Notes
    -----
    same result as Stata in UCLA example and same as Peterson

    """
    xu, hessian_inv = _get_sandwich_arrays(results, cov_type='clu')
    if not hasattr(group, 'dtype') or group.dtype != np.dtype('int'):
        clusters, group = np.unique(group, return_inverse=True)
    else:
        clusters = np.unique(group)
    scale = S_crosssection(xu, group)
    nobs, k_params = xu.shape
    n_groups = len(clusters)
    cov_c = _HCCM2(hessian_inv, scale)
    if use_correction:
        cov_c *= n_groups / (n_groups - 1.0) * ((nobs - 1.0) / float(nobs - k_params))
    return cov_c