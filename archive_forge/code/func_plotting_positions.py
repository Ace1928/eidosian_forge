import numpy as np
from numpy import ndarray
import numpy.ma as ma
from numpy.ma import masked, nomask
import math
import itertools
import warnings
from collections import namedtuple
from . import distributions
from scipy._lib._util import _rename_parameter, _contains_nan
from scipy._lib._bunch import _make_tuple_bunch
import scipy.special as special
import scipy.stats._stats_py
from ._stats_mstats_common import (
def plotting_positions(data, alpha=0.4, beta=0.4):
    """
    Returns plotting positions (or empirical percentile points) for the data.

    Plotting positions are defined as ``(i-alpha)/(n+1-alpha-beta)``, where:
        - i is the rank order statistics
        - n is the number of unmasked values along the given axis
        - `alpha` and `beta` are two parameters.

    Typical values for `alpha` and `beta` are:
        - (0,1)    : ``p(k) = k/n``, linear interpolation of cdf (R, type 4)
        - (.5,.5)  : ``p(k) = (k-1/2.)/n``, piecewise linear function
          (R, type 5)
        - (0,0)    : ``p(k) = k/(n+1)``, Weibull (R type 6)
        - (1,1)    : ``p(k) = (k-1)/(n-1)``, in this case,
          ``p(k) = mode[F(x[k])]``. That's R default (R type 7)
        - (1/3,1/3): ``p(k) = (k-1/3)/(n+1/3)``, then
          ``p(k) ~ median[F(x[k])]``.
          The resulting quantile estimates are approximately median-unbiased
          regardless of the distribution of x. (R type 8)
        - (3/8,3/8): ``p(k) = (k-3/8)/(n+1/4)``, Blom.
          The resulting quantile estimates are approximately unbiased
          if x is normally distributed (R type 9)
        - (.4,.4)  : approximately quantile unbiased (Cunnane)
        - (.35,.35): APL, used with PWM
        - (.3175, .3175): used in scipy.stats.probplot

    Parameters
    ----------
    data : array_like
        Input data, as a sequence or array of dimension at most 2.
    alpha : float, optional
        Plotting positions parameter. Default is 0.4.
    beta : float, optional
        Plotting positions parameter. Default is 0.4.

    Returns
    -------
    positions : MaskedArray
        The calculated plotting positions.

    """
    data = ma.array(data, copy=False).reshape(1, -1)
    n = data.count()
    plpos = np.empty(data.size, dtype=float)
    plpos[n:] = 0
    plpos[data.argsort(axis=None)[:n]] = (np.arange(1, n + 1) - alpha) / (n + 1.0 - alpha - beta)
    return ma.array(plpos, mask=data._mask)