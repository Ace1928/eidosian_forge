from statsmodels.compat.pandas import deprecate_kwarg
from collections.abc import Iterable
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper
from statsmodels.stats._adnorm import anderson_statistic, normal_ad
from statsmodels.stats._lilliefors import (
from statsmodels.tools.validation import (
from statsmodels.tsa.tsatools import lagmat
def breaks_hansen(olsresults):
    """
    Test for model stability, breaks in parameters for ols, Hansen 1992

    Parameters
    ----------
    olsresults : RegressionResults
        Results from estimation of a regression model.

    Returns
    -------
    teststat : float
        Hansen's test statistic.
    crit : ndarray
        The critical values at alpha=0.95 for different nvars.

    Notes
    -----
    looks good in example, maybe not very powerful for small changes in
    parameters

    According to Greene, distribution of test statistics depends on nvar but
    not on nobs.

    Test statistic is verified against R:strucchange

    References
    ----------
    Greene section 7.5.1, notation follows Greene
    """
    x = olsresults.model.exog
    resid = array_like(olsresults.resid, 'resid', shape=(x.shape[0], 1))
    nobs, nvars = x.shape
    resid2 = resid ** 2
    ft = np.c_[x * resid[:, None], resid2 - resid2.mean()]
    score = ft.cumsum(0)
    f = nobs * (ft[:, :, None] * ft[:, None, :]).sum(0)
    s = (score[:, :, None] * score[:, None, :]).sum(0)
    h = np.trace(np.dot(np.linalg.inv(f), s))
    crit95 = np.array([(2, 1.01), (6, 1.9), (15, 3.75), (19, 4.52)], dtype=[('nobs', int), ('crit', float)])
    return (h, crit95)