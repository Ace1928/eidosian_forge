import warnings
from numbers import Real
import numpy as np
from scipy.special import xlogy
from ..exceptions import UndefinedMetricWarning
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.stats import _weighted_percentile
from ..utils.validation import (
@validate_params({'y_true': ['array-like'], 'y_pred': ['array-like'], 'sample_weight': ['array-like', None], 'power': [Interval(Real, None, 0, closed='right'), Interval(Real, 1, None, closed='left')]}, prefer_skip_nested_validation=True)
def d2_tweedie_score(y_true, y_pred, *, sample_weight=None, power=0):
    """
    :math:`D^2` regression score function, fraction of Tweedie deviance explained.

    Best possible score is 1.0 and it can be negative (because the model can be
    arbitrarily worse). A model that always uses the empirical mean of `y_true` as
    constant prediction, disregarding the input features, gets a D^2 score of 0.0.

    Read more in the :ref:`User Guide <d2_score>`.

    .. versionadded:: 1.0

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    power : float, default=0
        Tweedie power parameter. Either power <= 0 or power >= 1.

        The higher `p` the less weight is given to extreme
        deviations between true and predicted targets.

        - power < 0: Extreme stable distribution. Requires: y_pred > 0.
        - power = 0 : Normal distribution, output corresponds to r2_score.
          y_true and y_pred can be any real numbers.
        - power = 1 : Poisson distribution. Requires: y_true >= 0 and
          y_pred > 0.
        - 1 < p < 2 : Compound Poisson distribution. Requires: y_true >= 0
          and y_pred > 0.
        - power = 2 : Gamma distribution. Requires: y_true > 0 and y_pred > 0.
        - power = 3 : Inverse Gaussian distribution. Requires: y_true > 0
          and y_pred > 0.
        - otherwise : Positive stable distribution. Requires: y_true > 0
          and y_pred > 0.

    Returns
    -------
    z : float or ndarray of floats
        The D^2 score.

    Notes
    -----
    This is not a symmetric function.

    Like R^2, D^2 score may be negative (it need not actually be the square of
    a quantity D).

    This metric is not well-defined for single samples and will return a NaN
    value if n_samples is less than two.

    References
    ----------
    .. [1] Eq. (3.11) of Hastie, Trevor J., Robert Tibshirani and Martin J.
           Wainwright. "Statistical Learning with Sparsity: The Lasso and
           Generalizations." (2015). https://hastie.su.domains/StatLearnSparsity/

    Examples
    --------
    >>> from sklearn.metrics import d2_tweedie_score
    >>> y_true = [0.5, 1, 2.5, 7]
    >>> y_pred = [1, 1, 5, 3.5]
    >>> d2_tweedie_score(y_true, y_pred)
    0.285...
    >>> d2_tweedie_score(y_true, y_pred, power=1)
    0.487...
    >>> d2_tweedie_score(y_true, y_pred, power=2)
    0.630...
    >>> d2_tweedie_score(y_true, y_true, power=2)
    1.0
    """
    y_type, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, None, dtype=[np.float64, np.float32])
    if y_type == 'continuous-multioutput':
        raise ValueError('Multioutput not supported in d2_tweedie_score')
    if _num_samples(y_pred) < 2:
        msg = 'D^2 score is not well-defined with less than two samples.'
        warnings.warn(msg, UndefinedMetricWarning)
        return float('nan')
    y_true, y_pred = (np.squeeze(y_true), np.squeeze(y_pred))
    numerator = mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=power)
    y_avg = np.average(y_true, weights=sample_weight)
    denominator = _mean_tweedie_deviance(y_true, y_avg, sample_weight=sample_weight, power=power)
    return 1 - numerator / denominator