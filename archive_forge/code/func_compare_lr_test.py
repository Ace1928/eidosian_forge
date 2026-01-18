from __future__ import annotations
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
from typing import Literal
from collections.abc import Sequence
import warnings
import numpy as np
from scipy import optimize, stats
from scipy.linalg import cholesky, toeplitz
from scipy.linalg.lapack import dtrtri
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.emplike.elregress import _ELRegOpts
from statsmodels.regression._prediction import PredictionResults
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import pinv_extended
from statsmodels.tools.typing import Float64Array
from statsmodels.tools.validation import bool_like, float_like, string_like
from . import _prediction as pred
def compare_lr_test(self, restricted, large_sample=False):
    """
        Likelihood ratio test to test whether restricted model is correct.

        Parameters
        ----------
        restricted : Result instance
            The restricted model is assumed to be nested in the current model.
            The result instance of the restricted model is required to have two
            attributes, residual sum of squares, `ssr`, residual degrees of
            freedom, `df_resid`.

        large_sample : bool
            Flag indicating whether to use a heteroskedasticity robust version
            of the LR test, which is a modified LM test.

        Returns
        -------
        lr_stat : float
            The likelihood ratio which is chisquare distributed with df_diff
            degrees of freedom.
        p_value : float
            The p-value of the test statistic.
        df_diff : int
            The degrees of freedom of the restriction, i.e. difference in df
            between models.

        Notes
        -----
        The exact likelihood ratio is valid for homoskedastic data,
        and is defined as

        .. math:: D=-2\\log\\left(\\frac{\\mathcal{L}_{null}}
           {\\mathcal{L}_{alternative}}\\right)

        where :math:`\\mathcal{L}` is the likelihood of the
        model. With :math:`D` distributed as chisquare with df equal
        to difference in number of parameters or equivalently
        difference in residual degrees of freedom.

        The large sample version of the likelihood ratio is defined as

        .. math:: D=n s^{\\prime}S^{-1}s

        where :math:`s=n^{-1}\\sum_{i=1}^{n} s_{i}`

        .. math:: s_{i} = x_{i,alternative} \\epsilon_{i,null}

        is the average score of the model evaluated using the
        residuals from null model and the regressors from the
        alternative model and :math:`S` is the covariance of the
        scores, :math:`s_{i}`.  The covariance of the scores is
        estimated using the same estimator as in the alternative
        model.

        This test compares the loglikelihood of the two models.  This
        may not be a valid test, if there is unspecified
        heteroscedasticity or correlation. This method will issue a
        warning if this is detected but still return the results
        without taking unspecified heteroscedasticity or correlation
        into account.

        This test compares the loglikelihood of the two models.  This
        may not be a valid test, if there is unspecified
        heteroscedasticity or correlation. This method will issue a
        warning if this is detected but still return the results
        without taking unspecified heteroscedasticity or correlation
        into account.

        is the average score of the model evaluated using the
        residuals from null model and the regressors from the
        alternative model and :math:`S` is the covariance of the
        scores, :math:`s_{i}`.  The covariance of the scores is
        estimated using the same estimator as in the alternative
        model.
        """
    if large_sample:
        return self.compare_lm_test(restricted, use_lr=True)
    has_robust1 = getattr(self, 'cov_type', 'nonrobust') != 'nonrobust'
    has_robust2 = getattr(restricted, 'cov_type', 'nonrobust') != 'nonrobust'
    if has_robust1 or has_robust2:
        warnings.warn('Likelihood Ratio test is likely invalid with ' + 'robust covariance, proceeding anyway', InvalidTestWarning)
    llf_full = self.llf
    llf_restr = restricted.llf
    df_full = self.df_resid
    df_restr = restricted.df_resid
    lrdf = df_restr - df_full
    lrstat = -2 * (llf_restr - llf_full)
    lr_pvalue = stats.chi2.sf(lrstat, lrdf)
    return (lrstat, lr_pvalue, lrdf)