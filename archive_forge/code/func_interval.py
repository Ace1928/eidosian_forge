from scipy._lib._util import getfullargspec_no_self as _getfullargspec
import sys
import keyword
import re
import types
import warnings
from itertools import zip_longest
from scipy._lib import doccer
from ._distr_params import distcont, distdiscrete
from scipy._lib._util import check_random_state
from scipy.special import comb, entr
from scipy import optimize
from scipy import integrate
from scipy._lib._finite_differences import _derivative
from scipy import stats
from numpy import (arange, putmask, ones, shape, ndarray, zeros, floor,
import numpy as np
from ._constants import _XMAX, _LOGXMAX
from ._censored_data import CensoredData
from scipy.stats._warnings_errors import FitError
def interval(self, confidence, *args, **kwds):
    """Confidence interval with equal areas around the median.

        Parameters
        ----------
        confidence : array_like of float
            Probability that an rv will be drawn from the returned range.
            Each value should be in the range [0, 1].
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            location parameter, Default is 0.
        scale : array_like, optional
            scale parameter, Default is 1.

        Returns
        -------
        a, b : ndarray of float
            end-points of range that contain ``100 * alpha %`` of the rv's
            possible values.

        Notes
        -----
        This is implemented as ``ppf([p_tail, 1-p_tail])``, where
        ``ppf`` is the inverse cumulative distribution function and
        ``p_tail = (1-confidence)/2``. Suppose ``[c, d]`` is the support of a
        discrete distribution; then ``ppf([0, 1]) == (c-1, d)``. Therefore,
        when ``confidence=1`` and the distribution is discrete, the left end
        of the interval will be beyond the support of the distribution.
        For discrete distributions, the interval will limit the probability
        in each tail to be less than or equal to ``p_tail`` (usually
        strictly less).

        """
    alpha = confidence
    alpha = asarray(alpha)
    if np.any((alpha > 1) | (alpha < 0)):
        raise ValueError('alpha must be between 0 and 1 inclusive')
    q1 = (1.0 - alpha) / 2
    q2 = (1.0 + alpha) / 2
    a = self.ppf(q1, *args, **kwds)
    b = self.ppf(q2, *args, **kwds)
    return (a, b)