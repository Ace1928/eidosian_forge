import warnings # for silencing, see above...
import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats, special
from statsmodels.sandbox.distributions.extras import (
def f_stats(self, dfn, dfd):
    arr, where, inf, sqrt, nan = (np.array, np.where, np.inf, np.sqrt, np.nan)
    v2 = arr(dfd * 1.0)
    v1 = arr(dfn * 1.0)
    mu = where(v2 > 2, v2 / arr(v2 - 2), inf)
    mu2 = 2 * v2 * v2 * (v2 + v1 - 2) / (v1 * (v2 - 2) ** 2 * (v2 - 4))
    mu2 = where(v2 > 4, mu2, inf)
    g1 = 2 * (v2 + 2 * v1 - 2.0) / (v2 - 6.0) * np.sqrt(2 * (v2 - 4.0) / (v1 * (v2 + v1 - 2.0)))
    g1 = where(v2 > 6, g1, nan)
    g2 = 3 / (2.0 * v2 - 16) * (8 + g1 * g1 * (v2 - 6.0))
    g2 = where(v2 > 8, g2, nan)
    return (mu, mu2, g1, g2)