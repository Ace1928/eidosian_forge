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
def compare_encompassing(results_x, results_z, cov_type='nonrobust', cov_kwargs=None):
    """
    Davidson-MacKinnon encompassing test for comparing non-nested models

    Parameters
    ----------
    results_x : Result instance
        result instance of first model
    results_z : Result instance
        result instance of second model
    cov_type : str, default "nonrobust
        Covariance type. The default is "nonrobust` which uses the classic
        OLS covariance estimator. Specify one of "HC0", "HC1", "HC2", "HC3"
        to use White's covariance estimator. All covariance types supported
        by ``OLS.fit`` are accepted.
    cov_kwargs : dict, default None
        Dictionary of covariance options passed to ``OLS.fit``. See OLS.fit
        for more details.

    Returns
    -------
    DataFrame
        A DataFrame with two rows and four columns. The row labeled x
        contains results for the null that the model contained in
        results_x is equivalent to the encompassing model. The results in
        the row labeled z correspond to the test that the model contained
        in results_z are equivalent to the encompassing model. The columns
        are the test statistic, its p-value, and the numerator and
        denominator degrees of freedom. The test statistic has an F
        distribution. The numerator degree of freedom is the number of
        variables in the encompassing model that are not in the x or z model.
        The denominator degree of freedom is the number of observations minus
        the number of variables in the nesting model.

    Notes
    -----
    The null is that the fit produced using x is the same as the fit
    produced using both x and z. When testing whether x is encompassed,
    the model estimated is

    .. math::

        Y = X\\beta + Z_1\\gamma + \\epsilon

    where :math:`Z_1` are the columns of :math:`Z` that are not spanned by
    :math:`X`. The null is :math:`H_0:\\gamma=0`. When testing whether z is
    encompassed, the roles of :math:`X` and :math:`Z` are reversed.

    Implementation of  Davidson and MacKinnon (1993)'s encompassing test.
    Performs two Wald tests where models x and z are compared to a model
    that nests the two. The Wald tests are performed by using an OLS
    regression.
    """
    if _check_nested_results(results_x, results_z):
        raise ValueError(NESTED_ERROR.format(test='Testing encompassing'))
    y = results_x.model.endog
    x = results_x.model.exog
    z = results_z.model.exog

    def _test_nested(endog, a, b, cov_est, cov_kwds):
        err = b - a @ np.linalg.lstsq(a, b, rcond=None)[0]
        u, s, v = np.linalg.svd(err)
        eps = np.finfo(np.double).eps
        tol = s.max(axis=-1, keepdims=True) * max(err.shape) * eps
        non_zero = np.abs(s) > tol
        aug = err @ v[:, non_zero]
        aug_reg = np.hstack([a, aug])
        k_a = aug.shape[1]
        k = aug_reg.shape[1]
        res = OLS(endog, aug_reg).fit(cov_type=cov_est, cov_kwds=cov_kwds)
        r_matrix = np.zeros((k_a, k))
        r_matrix[:, -k_a:] = np.eye(k_a)
        test = res.wald_test(r_matrix, use_f=True, scalar=True)
        stat, pvalue = (test.statistic, test.pvalue)
        df_num, df_denom = (int(test.df_num), int(test.df_denom))
        return (stat, pvalue, df_num, df_denom)
    x_nested = _test_nested(y, x, z, cov_type, cov_kwargs)
    z_nested = _test_nested(y, z, x, cov_type, cov_kwargs)
    return pd.DataFrame([x_nested, z_nested], index=['x', 'z'], columns=['stat', 'pvalue', 'df_num', 'df_denom'])