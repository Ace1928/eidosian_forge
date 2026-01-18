import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def compute_smoothed_state_weights(results, compute_t=None, compute_j=None, compute_prior_weights=None, resmooth=None):
    """
    Construct the weights of observations and the prior on the smoothed state

    Parameters
    ----------
    results : MLEResults object
        Results object from fitting a state space model.
    compute_t : array_like, optional
        An explicit list of periods `t` of the smoothed state vector to compute
        weights for (see the Returns section for more details about the
        dimension `t`). Default is to compute weights for all periods `t`.
        However, if weights for only a few time points are desired, then
        performance can be improved by specifying this argument.
    compute_j : array_like, optional
        An explicit list of periods `j` of observations to compute
        weights for (see the Returns section for more details about the
        dimension `j`). Default is to compute weights for all periods `j`.
        However, if weights for only a few time points are desired, then
        performance can be improved by specifying this argument.
    compute_prior_weights : bool, optional
        Whether or not to compute the weight matrices associated with the prior
        mean (also called the "initial state"). Note that doing so requires
        that period 0 is in the periods defined in `compute_j`. Default is True
        if 0 is in `compute_j` (or if the `compute_j` argument is not passed)
        and False otherwise.
    resmooth : bool, optional
        Whether or not to re-perform filtering and smoothing prior to
        constructing the weights. Default is to resmooth if the smoothed_state
        vector is different between the given results object and the
        underlying smoother. Caution is adviced when changing this setting.
        See the Notes section below for more details.

    Returns
    -------
    weights : array_like
        Weight matrices that can be used to construct the smoothed state from
        the observations. The returned matrix is always shaped
        `(nobs, nobs, k_states, k_endog)`, and entries that are not computed
        are set to NaNs. (Entries will not be computed if they are not
        included in `compute_t` and `compute_j`, or if they correspond to
        missing observations, or if they are for periods in which the exact
        diffuse Kalman filter is operative). The `(t, j, m, p)`-th element of
        this matrix contains the weight of the `p`-th element of the
        observation vector at time `j` in constructing the `m`-th element of
        the smoothed state vector at time `t`.
    prior_weights : array_like
        Weight matrices that describe the impact of the prior (also called the
        initialization) on the smoothed state vector. The returned matrix is
        always shaped `(nobs, k_states, k_states)`. If prior weights are not
        computed, then all entries will be set to NaNs. The `(t, m, l)`-th
        element of this matrix contains the weight of the `l`-th element of the
        prior mean (also called the "initial state") in constructing the
        `m`-th element of the smoothed state vector at time `t`.

    Notes
    -----
    In [1]_, Chapter 4.8, it is shown how the smoothed state vector can be
    written as a weighted vector sum of observations:

    .. math::

        \\hat \\alpha_t = \\sum_{j=1}^n \\omega_{jt}^{\\hat \\alpha} y_j

    One output of this function is the weights
    :math:`\\omega_{jt}^{\\hat \\alpha}`. Note that the description in [1]_
    assumes that the prior mean (or "initial state") is fixed to be zero. More
    generally, the smoothed state vector will also depend partly on the prior.
    The second output of this function are the weights of the prior mean.

    There are two important technical notes about the computations used here:

    1. In the univariate approach to multivariate filtering (see e.g.
       Chapter 6.4 of [1]_), all observations are introduced one at a time,
       including those from the same time period. As a result, the weight of
       each observation can be different than when all observations from the
       same time point are introduced together, as in the typical multivariate
       filtering approach. Here, we always compute weights as in the
       multivariate filtering approach, and we handle singular forecast error
       covariance matrices by using a pseudo-inverse.
    2. Constructing observation weights for periods in which the exact diffuse
       filter (see e.g. Chapter 5 of [1]_) is operative is not done here, and
       so the corresponding entries in the returned weight matrices will always
       be set equal to zeros. While handling these periods may be implemented
       in the future, one option for constructing these weights is to use an
       approximate (instead of exact) diffuse initialization for this purpose.

    Finally, one note about implementation: to compute the weights, we use
    attributes of the underlying filtering and smoothing Cython objects
    directly. However, these objects are not frozen with the result
    computation, and we cannot guarantee that their attributes have not
    changed since `res` was created. As a result, by default we re-run the
    filter and smoother to ensure that the attributes there actually correspond
    to the `res` object. This can be overridden by the user for a small
    performance boost if they are sure that the attributes have not changed;
    see the `resmooth` argument.

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
            Time Series Analysis by State Space Methods: Second Edition.
            Oxford University Press.
    """
    mod = results.model
    mod.update(results.params)
    if resmooth is None:
        resmooth = np.any(results.smoothed_state != mod.ssm._kalman_smoother.smoothed_state)
    if resmooth:
        mod.ssm.smooth(conserve_memory=0, update_representation=False, update_filter=False, update_smoother=False)
    else:
        mod.ssm._initialize_representation()
    return _compute_smoothed_state_weights(mod.ssm, compute_t=compute_t, compute_j=compute_j, compute_prior_weights=compute_prior_weights, scale=results.filter_results.scale)