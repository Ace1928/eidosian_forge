import warnings
from numbers import Real
import numpy as np
from scipy.special import xlogy
from ..exceptions import UndefinedMetricWarning
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.stats import _weighted_percentile
from ..utils.validation import (
@validate_params({'y_true': ['array-like'], 'y_pred': ['array-like'], 'sample_weight': ['array-like', None], 'power': [Interval(Real, None, 0, closed='right'), Interval(Real, 1, None, closed='left')]}, prefer_skip_nested_validation=True)
def mean_tweedie_deviance(y_true, y_pred, *, sample_weight=None, power=0):
    """Mean Tweedie deviance regression loss.

    Read more in the :ref:`User Guide <mean_tweedie_deviance>`.

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
        - power = 0 : Normal distribution, output corresponds to
          mean_squared_error. y_true and y_pred can be any real numbers.
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
    loss : float
        A non-negative floating point value (the best value is 0.0).

    Examples
    --------
    >>> from sklearn.metrics import mean_tweedie_deviance
    >>> y_true = [2, 0, 1, 4]
    >>> y_pred = [0.5, 0.5, 2., 2.]
    >>> mean_tweedie_deviance(y_true, y_pred, power=1)
    1.4260...
    """
    y_type, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, None, dtype=[np.float64, np.float32])
    if y_type == 'continuous-multioutput':
        raise ValueError('Multioutput not supported in mean_tweedie_deviance')
    check_consistent_length(y_true, y_pred, sample_weight)
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        sample_weight = sample_weight[:, np.newaxis]
    message = f'Mean Tweedie deviance error with power={power} can only be used on '
    if power < 0:
        if (y_pred <= 0).any():
            raise ValueError(message + 'strictly positive y_pred.')
    elif power == 0:
        pass
    elif 1 <= power < 2:
        if (y_true < 0).any() or (y_pred <= 0).any():
            raise ValueError(message + 'non-negative y and strictly positive y_pred.')
    elif power >= 2:
        if (y_true <= 0).any() or (y_pred <= 0).any():
            raise ValueError(message + 'strictly positive y and y_pred.')
    else:
        raise ValueError
    return _mean_tweedie_deviance(y_true, y_pred, sample_weight=sample_weight, power=power)