import numpy as np
from scipy import stats
def gbic(results, gbicp=False):
    """generalized BIC for misspecified models

    References
    ----------
    Lv, Jinchi, and Jun S. Liu. 2014. "Model Selection Principles in
    Misspecified Models." Journal of the Royal Statistical Society.
    Series B (Statistical Methodology) 76 (1): 141–67.

    """
    self = getattr(results, '_results', results)
    k_params = self.df_model + 1
    nobs = k_params + self.df_resid
    imr = getattr(results, 'im_ratio', im_ratio(results))
    imr_logdet = np.linalg.slogdet(imr)[1]
    gbic = -2 * self.llf + k_params * np.log(nobs) - imr_logdet
    gbicp = gbic + np.trace(imr)
    return (gbic, gbicp)