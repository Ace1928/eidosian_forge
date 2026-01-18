import warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, bic, hqic
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tools.tools import Bunch, pinv_extended
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.regime_switching._hamilton_filter import (
from statsmodels.tsa.regime_switching._kim_smoother import (
from statsmodels.tsa.statespace.tools import (
def cy_kim_smoother_log(regime_transition, predicted_joint_probabilities, filtered_joint_probabilities):
    """
    Kim smoother in log space using Cython inner loop.

    Parameters
    ----------
    regime_transition : ndarray
        Matrix of regime transition probabilities, shaped either
        (k_regimes, k_regimes, 1) or if there are time-varying transition
        probabilities (k_regimes, k_regimes, nobs).
    predicted_joint_probabilities : ndarray
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t-1}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t-1
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).
    filtered_joint_probabilities : ndarray
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).

    Returns
    -------
    smoothed_joint_probabilities : ndarray
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_T] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on all information.
        Shaped (k_regimes,) * (order + 1) + (nobs,).
    smoothed_marginal_probabilities : ndarray
        Array containing Pr[S_t=s_t | Y_T] - the probability of being in each
        regime conditional on all information. Shaped (k_regimes, nobs).
    """
    k_regimes = filtered_joint_probabilities.shape[0]
    nobs = filtered_joint_probabilities.shape[-1]
    order = filtered_joint_probabilities.ndim - 2
    dtype = filtered_joint_probabilities.dtype
    smoothed_joint_probabilities = np.zeros((k_regimes,) * (order + 1) + (nobs,), dtype=dtype)
    if regime_transition.shape[-1] == nobs + order:
        regime_transition = regime_transition[..., order:]
    regime_transition = np.log(np.maximum(regime_transition, 1e-20))
    prefix, dtype, _ = find_best_blas_type((regime_transition, predicted_joint_probabilities, filtered_joint_probabilities))
    func = prefix_kim_smoother_log_map[prefix]
    func(nobs, k_regimes, order, regime_transition, predicted_joint_probabilities.reshape(k_regimes ** (order + 1), nobs), filtered_joint_probabilities.reshape(k_regimes ** (order + 1), nobs), smoothed_joint_probabilities.reshape(k_regimes ** (order + 1), nobs))
    smoothed_joint_probabilities = np.exp(smoothed_joint_probabilities)
    smoothed_marginal_probabilities = smoothed_joint_probabilities
    for i in range(1, smoothed_marginal_probabilities.ndim - 1):
        smoothed_marginal_probabilities = np.sum(smoothed_marginal_probabilities, axis=-2)
    return (smoothed_joint_probabilities, smoothed_marginal_probabilities)