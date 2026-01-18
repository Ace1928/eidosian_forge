import numpy as np
from scipy import stats
from scipy.special import comb
import warnings
from statsmodels.tools.validation import array_like
def runs_test(self, correction=True):
    """basic version of runs test

        Parameters
        ----------
        correction : bool
            Following the SAS manual, for samplesize below 50, the test
            statistic is corrected by 0.5. This can be turned off with
            correction=False, and was included to match R, tseries, which
            does not use any correction.

        pvalue based on normal distribution, with integer correction

        """
    self.npo = npo = self.runs_pos.sum()
    self.nne = nne = self.runs_neg.sum()
    n = npo + nne
    npn = npo * nne
    rmean = 2.0 * npn / n + 1
    rvar = 2.0 * npn * (2.0 * npn - n) / n ** 2.0 / (n - 1.0)
    rstd = np.sqrt(rvar)
    rdemean = self.n_runs - rmean
    if n >= 50 or not correction:
        z = rdemean
    elif rdemean > 0.5:
        z = rdemean - 0.5
    elif rdemean < 0.5:
        z = rdemean + 0.5
    else:
        z = 0.0
    z /= rstd
    pval = 2 * stats.norm.sf(np.abs(z))
    return (z, pval)