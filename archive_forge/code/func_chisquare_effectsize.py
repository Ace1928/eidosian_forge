from statsmodels.compat.python import lrange
import numpy as np
from scipy import stats
def chisquare_effectsize(probs0, probs1, correction=None, cohen=True, axis=0):
    """effect size for a chisquare goodness-of-fit test

    Parameters
    ----------
    probs0 : array_like
        probabilities or cell frequencies under the Null hypothesis
    probs1 : array_like
        probabilities or cell frequencies under the Alternative hypothesis
        probs0 and probs1 need to have the same length in the ``axis`` dimension.
        and broadcast in the other dimensions
        Both probs0 and probs1 are normalized to add to one (in the ``axis``
        dimension).
    correction : None or tuple
        If None, then the effect size is the chisquare statistic divide by
        the number of observations.
        If the correction is a tuple (nobs, df), then the effectsize is
        corrected to have less bias and a smaller variance. However, the
        correction can make the effectsize negative. In that case, the
        effectsize is set to zero.
        Pederson and Johnson (1990) as referenced in McLaren et all. (1994)
    cohen : bool
        If True, then the square root is returned as in the definition of the
        effect size by Cohen (1977), If False, then the original effect size
        is returned.
    axis : int
        If the probability arrays broadcast to more than 1 dimension, then
        this is the axis over which the sums are taken.

    Returns
    -------
    effectsize : float
        effect size of chisquare test

    """
    probs0 = np.asarray(probs0, float)
    probs1 = np.asarray(probs1, float)
    probs0 = probs0 / probs0.sum(axis)
    probs1 = probs1 / probs1.sum(axis)
    d2 = ((probs1 - probs0) ** 2 / probs0).sum(axis)
    if correction is not None:
        nobs, df = correction
        diff = ((probs1 - probs0) / probs0).sum(axis)
        d2 = np.maximum((d2 * nobs - diff - df) / (nobs - 1.0), 0)
    if cohen:
        return np.sqrt(d2)
    else:
        return d2