from statsmodels.compat.python import lmap
import os
import numpy as np
from scipy import stats
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from .try_ols_anova import data2dummy
def anova_oneway(y, x, seq=0):
    yrvs = y[:, np.newaxis]
    xrvs = x[:, np.newaxis] - x.mean()
    from .try_catdata import groupsstats_dummy
    meang, varg, xdevmeangr, countg = groupsstats_dummy(yrvs[:, :1], xrvs[:, :1])
    sswn = np.dot(xdevmeangr.T, xdevmeangr)
    ssbn = np.dot((meang - xrvs.mean()) ** 2, countg.T)
    nobs = yrvs.shape[0]
    ncat = meang.shape[1]
    dfbn = ncat - 1
    dfwn = nobs - ncat
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    f = msb / msw
    prob = stats.f.sf(f, dfbn, dfwn)
    R2 = ssbn / (sswn + ssbn)
    resstd = np.sqrt(msw)

    def _fix2scalar(z):
        if np.shape(z) == (1, 1):
            return z[0, 0]
        else:
            return z
    f, prob, R2, resstd = lmap(_fix2scalar, (f, prob, R2, resstd))
    return (f, prob, R2, resstd)