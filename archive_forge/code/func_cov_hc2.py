import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def cov_hc2(results):
    """
    See statsmodels.RegressionResults
    """
    h = np.diag(np.dot(results.model.exog, np.dot(results.normalized_cov_params, results.model.exog.T)))
    het_scale = results.resid ** 2 / (1 - h)
    cov_hc2_ = _HCCM(results, het_scale)
    return cov_hc2_