from __future__ import annotations
import math
import warnings
from collections import namedtuple
import numpy as np
from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
from scipy import optimize, special, interpolate, stats
from scipy._lib._bunch import _make_tuple_bunch
from scipy._lib._util import _rename_parameter, _contains_nan, _get_nan
from ._ansari_swilk_statistics import gscale, swilk
from . import _stats_py
from ._fit import FitResult
from ._stats_py import find_repeats, _normtest_finish, SignificanceResult
from .contingency import chi2_contingency
from . import distributions
from ._distn_infrastructure import rv_generic
from ._hypotests import _get_wilcoxon_distr
from ._axis_nan_policy import _axis_nan_policy_factory
@_axis_nan_policy_factory(lambda x: x, n_outputs=1, default_axis=None, result_to_tuple=lambda x: (x,))
def circmean(samples, high=2 * pi, low=0, axis=None, nan_policy='propagate'):
    """Compute the circular mean for samples in a range.

    Parameters
    ----------
    samples : array_like
        Input array.
    high : float or int, optional
        High boundary for the sample range. Default is ``2*pi``.
    low : float or int, optional
        Low boundary for the sample range. Default is 0.

    Returns
    -------
    circmean : float
        Circular mean.

    See Also
    --------
    circstd : Circular standard deviation.
    circvar : Circular variance.

    Examples
    --------
    For simplicity, all angles are printed out in degrees.

    >>> import numpy as np
    >>> from scipy.stats import circmean
    >>> import matplotlib.pyplot as plt
    >>> angles = np.deg2rad(np.array([20, 30, 330]))
    >>> circmean = circmean(angles)
    >>> np.rad2deg(circmean)
    7.294976657784009

    >>> mean = angles.mean()
    >>> np.rad2deg(mean)
    126.66666666666666

    Plot and compare the circular mean against the arithmetic mean.

    >>> plt.plot(np.cos(np.linspace(0, 2*np.pi, 500)),
    ...          np.sin(np.linspace(0, 2*np.pi, 500)),
    ...          c='k')
    >>> plt.scatter(np.cos(angles), np.sin(angles), c='k')
    >>> plt.scatter(np.cos(circmean), np.sin(circmean), c='b',
    ...             label='circmean')
    >>> plt.scatter(np.cos(mean), np.sin(mean), c='r', label='mean')
    >>> plt.legend()
    >>> plt.axis('equal')
    >>> plt.show()

    """
    samples, sin_samp, cos_samp = _circfuncs_common(samples, high, low)
    sin_sum = sin_samp.sum(axis)
    cos_sum = cos_samp.sum(axis)
    res = arctan2(sin_sum, cos_sum)
    res = np.asarray(res)
    res[res < 0] += 2 * pi
    res = res[()]
    return res * (high - low) / 2.0 / pi + low