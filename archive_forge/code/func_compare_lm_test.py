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
def compare_lm_test(self, restricted, demean=True, use_lr=False):
    """
        Use Lagrange Multiplier test to test a set of linear restrictions.

        Parameters
        ----------
        restricted : Result instance
            The restricted model is assumed to be nested in the
            current model. The result instance of the restricted model
            is required to have two attributes, residual sum of
            squares, `ssr`, residual degrees of freedom, `df_resid`.
        demean : bool
            Flag indicating whether the demean the scores based on the
            residuals from the restricted model.  If True, the covariance of
            the scores are used and the LM test is identical to the large
            sample version of the LR test.
        use_lr : bool
            A flag indicating whether to estimate the covariance of the model
            scores using the unrestricted model. Setting the to True improves
            the power of the test.

        Returns
        -------
        lm_value : float
            The test statistic which has a chi2 distributed.
        p_value : float
            The p-value of the test statistic.
        df_diff : int
            The degrees of freedom of the restriction, i.e. difference in df
            between models.

        Notes
        -----
        The LM test examines whether the scores from the restricted model are
        0. If the null is true, and the restrictions are valid, then the
        parameters of the restricted model should be close to the minimum of
        the sum of squared errors, and so the scores should be close to zero,
        on average.
        """
    from numpy.linalg import inv
    import statsmodels.stats.sandwich_covariance as sw
    if not self._is_nested(restricted):
        raise ValueError('Restricted model is not nested by full model.')
    wresid = restricted.wresid
    wexog = self.model.wexog
    scores = wexog * wresid[:, None]
    n = self.nobs
    df_full = self.df_resid
    df_restr = restricted.df_resid
    df_diff = df_restr - df_full
    s = scores.mean(axis=0)
    if use_lr:
        scores = wexog * self.wresid[:, None]
        demean = False
    if demean:
        scores = scores - scores.mean(0)[None, :]
    cov_type = getattr(self, 'cov_type', 'nonrobust')
    if cov_type == 'nonrobust':
        sigma2 = np.mean(wresid ** 2)
        xpx = np.dot(wexog.T, wexog) / n
        s_inv = inv(sigma2 * xpx)
    elif cov_type in ('HC0', 'HC1', 'HC2', 'HC3'):
        s_inv = inv(np.dot(scores.T, scores) / n)
    elif cov_type == 'HAC':
        maxlags = self.cov_kwds['maxlags']
        s_inv = inv(sw.S_hac_simple(scores, maxlags) / n)
    elif cov_type == 'cluster':
        groups = self.cov_kwds['groups']
        s_inv = inv(sw.S_crosssection(scores, groups))
    else:
        raise ValueError('Only nonrobust, HC, HAC and cluster are ' + 'currently connected')
    lm_value = n * (s @ s_inv @ s.T)
    p_value = stats.chi2.sf(lm_value, df_diff)
    return (lm_value, p_value, df_diff)