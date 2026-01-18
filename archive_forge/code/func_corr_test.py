import numpy as np
from numpy.linalg import svd
import scipy
import pandas as pd
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from .multivariate_ols import multivariate_stats
def corr_test(self):
    """Approximate F test
        Perform multivariate statistical tests of the hypothesis that
        there is no canonical correlation between endog and exog.
        For each canonical correlation, testing its significance based on
        Wilks' lambda.

        Returns
        -------
        CanCorrTestResults instance
        """
    nobs, k_yvar = self.endog.shape
    nobs, k_xvar = self.exog.shape
    eigenvals = np.power(self.cancorr, 2)
    stats = pd.DataFrame(columns=['Canonical Correlation', "Wilks' lambda", 'Num DF', 'Den DF', 'F Value', 'Pr > F'], index=list(range(len(eigenvals) - 1, -1, -1)))
    prod = 1
    for i in range(len(eigenvals) - 1, -1, -1):
        prod *= 1 - eigenvals[i]
        p = k_yvar - i
        q = k_xvar - i
        r = nobs - k_yvar - 1 - (p - q + 1) / 2
        u = (p * q - 2) / 4
        df1 = p * q
        if p ** 2 + q ** 2 - 5 > 0:
            t = np.sqrt(((p * q) ** 2 - 4) / (p ** 2 + q ** 2 - 5))
        else:
            t = 1
        df2 = r * t - 2 * u
        lmd = np.power(prod, 1 / t)
        F = (1 - lmd) / lmd * df2 / df1
        stats.loc[i, 'Canonical Correlation'] = self.cancorr[i]
        stats.loc[i, "Wilks' lambda"] = prod
        stats.loc[i, 'Num DF'] = df1
        stats.loc[i, 'Den DF'] = df2
        stats.loc[i, 'F Value'] = F
        pval = scipy.stats.f.sf(F, df1, df2)
        stats.loc[i, 'Pr > F'] = pval
        "\n            # Wilk's Chi square test of each canonical correlation\n            df = (p - i + 1) * (q - i + 1)\n            chi2 = a * np.log(prod)\n            pval = stats.chi2.sf(chi2, df)\n            stats.loc[i, 'Canonical correlation'] = self.cancorr[i]\n            stats.loc[i, 'Chi-square'] = chi2\n            stats.loc[i, 'DF'] = df\n            stats.loc[i, 'Pr > ChiSq'] = pval\n            "
    ind = stats.index.values[::-1]
    stats = stats.loc[ind, :]
    stats_mv = multivariate_stats(eigenvals, k_yvar, k_xvar, nobs - k_xvar - 1)
    return CanCorrTestResults(stats, stats_mv)