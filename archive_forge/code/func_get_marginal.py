import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.distributions.tools import (
def get_marginal(self, idx):
    """Get marginal BernsteinDistribution.

        Parameters
        ----------
        idx : int or list of int
            Index or indices of the component for which the marginal
            distribution is returned.

        Returns
        -------
        BernsteinDistribution instance for the marginal distribution.
        """
    if self.k_dim == 1:
        return self
    sl = [-1] * self.k_dim
    if np.shape(idx) == ():
        idx = [idx]
    for ii in idx:
        sl[ii] = slice(None, None, None)
    cdf_m = self.cdf_grid[tuple(sl)]
    bpd_marginal = BernsteinDistribution(cdf_m)
    return bpd_marginal