import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def cov_cluster_2groups(results, group, group2=None, use_correction=True):
    """cluster robust covariance matrix for two groups/clusters

    Parameters
    ----------
    results : result instance
       result of a regression, uses results.model.exog and results.resid
       TODO: this should use wexog instead
    use_correction : bool
       If true (default), then the small sample correction factor is used.

    Returns
    -------
    cov_both : ndarray, (k_vars, k_vars)
        cluster robust covariance matrix for parameter estimates, for both
        clusters
    cov_0 : ndarray, (k_vars, k_vars)
        cluster robust covariance matrix for parameter estimates for first
        cluster
    cov_1 : ndarray, (k_vars, k_vars)
        cluster robust covariance matrix for parameter estimates for second
        cluster

    Notes
    -----

    verified against Peterson's table, (4 decimal print precision)
    """
    if group2 is None:
        if group.ndim != 2 or group.shape[1] != 2:
            raise ValueError('if group2 is not given, then groups needs to be ' + 'an array with two columns')
        group0 = group[:, 0]
        group1 = group[:, 1]
    else:
        group0 = group
        group1 = group2
        group = (group0, group1)
    cov0 = cov_cluster(results, group0, use_correction=use_correction)
    cov1 = cov_cluster(results, group1, use_correction=use_correction)
    cov01 = cov_cluster(results, combine_indices(group)[0], use_correction=use_correction)
    cov_both = cov0 + cov1 - cov01
    return (cov_both, cov0, cov1)