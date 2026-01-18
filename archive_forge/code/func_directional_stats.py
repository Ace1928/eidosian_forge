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
def directional_stats(samples, *, axis=0, normalize=True):
    """
    Computes sample statistics for directional data.

    Computes the directional mean (also called the mean direction vector) and
    mean resultant length of a sample of vectors.

    The directional mean is a measure of "preferred direction" of vector data.
    It is analogous to the sample mean, but it is for use when the length of
    the data is irrelevant (e.g. unit vectors).

    The mean resultant length is a value between 0 and 1 used to quantify the
    dispersion of directional data: the smaller the mean resultant length, the
    greater the dispersion. Several definitions of directional variance
    involving the mean resultant length are given in [1]_ and [2]_.

    Parameters
    ----------
    samples : array_like
        Input array. Must be at least two-dimensional, and the last axis of the
        input must correspond with the dimensionality of the vector space.
        When the input is exactly two dimensional, this means that each row
        of the data is a vector observation.
    axis : int, default: 0
        Axis along which the directional mean is computed.
    normalize: boolean, default: True
        If True, normalize the input to ensure that each observation is a
        unit vector. It the observations are already unit vectors, consider
        setting this to False to avoid unnecessary computation.

    Returns
    -------
    res : DirectionalStats
        An object containing attributes:

        mean_direction : ndarray
            Directional mean.
        mean_resultant_length : ndarray
            The mean resultant length [1]_.

    See Also
    --------
    circmean: circular mean; i.e. directional mean for 2D *angles*
    circvar: circular variance; i.e. directional variance for 2D *angles*

    Notes
    -----
    This uses a definition of directional mean from [1]_.
    Assuming the observations are unit vectors, the calculation is as follows.

    .. code-block:: python

        mean = samples.mean(axis=0)
        mean_resultant_length = np.linalg.norm(mean)
        mean_direction = mean / mean_resultant_length

    This definition is appropriate for *directional* data (i.e. vector data
    for which the magnitude of each observation is irrelevant) but not
    for *axial* data (i.e. vector data for which the magnitude and *sign* of
    each observation is irrelevant).

    Several definitions of directional variance involving the mean resultant
    length ``R`` have been proposed, including ``1 - R`` [1]_, ``1 - R**2``
    [2]_, and ``2 * (1 - R)`` [2]_. Rather than choosing one, this function
    returns ``R`` as attribute `mean_resultant_length` so the user can compute
    their preferred measure of dispersion.

    References
    ----------
    .. [1] Mardia, Jupp. (2000). *Directional Statistics*
       (p. 163). Wiley.

    .. [2] https://en.wikipedia.org/wiki/Directional_statistics

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import directional_stats
    >>> data = np.array([[3, 4],    # first observation, 2D vector space
    ...                  [6, -8]])  # second observation
    >>> dirstats = directional_stats(data)
    >>> dirstats.mean_direction
    array([1., 0.])

    In contrast, the regular sample mean of the vectors would be influenced
    by the magnitude of each observation. Furthermore, the result would not be
    a unit vector.

    >>> data.mean(axis=0)
    array([4.5, -2.])

    An exemplary use case for `directional_stats` is to find a *meaningful*
    center for a set of observations on a sphere, e.g. geographical locations.

    >>> data = np.array([[0.8660254, 0.5, 0.],
    ...                  [0.8660254, -0.5, 0.]])
    >>> dirstats = directional_stats(data)
    >>> dirstats.mean_direction
    array([1., 0., 0.])

    The regular sample mean on the other hand yields a result which does not
    lie on the surface of the sphere.

    >>> data.mean(axis=0)
    array([0.8660254, 0., 0.])

    The function also returns the mean resultant length, which
    can be used to calculate a directional variance. For example, using the
    definition ``Var(z) = 1 - R`` from [2]_ where ``R`` is the
    mean resultant length, we can calculate the directional variance of the
    vectors in the above example as:

    >>> 1 - dirstats.mean_resultant_length
    0.13397459716167093
    """
    samples = np.asarray(samples)
    if samples.ndim < 2:
        raise ValueError(f'samples must at least be two-dimensional. Instead samples has shape: {samples.shape!r}')
    samples = np.moveaxis(samples, axis, 0)
    if normalize:
        vectornorms = np.linalg.norm(samples, axis=-1, keepdims=True)
        samples = samples / vectornorms
    mean = np.mean(samples, axis=0)
    mean_resultant_length = np.linalg.norm(mean, axis=-1, keepdims=True)
    mean_direction = mean / mean_resultant_length
    return DirectionalStats(mean_direction, mean_resultant_length.squeeze(-1)[()])