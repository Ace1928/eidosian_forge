import warnings
from numbers import Real
import numpy as np
from scipy.special import xlogy
from ..exceptions import UndefinedMetricWarning
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.stats import _weighted_percentile
from ..utils.validation import (
@validate_params({'y_true': ['array-like'], 'y_pred': ['array-like'], 'sample_weight': ['array-like', None], 'multioutput': [StrOptions({'raw_values', 'uniform_average'}), 'array-like']}, prefer_skip_nested_validation=True)
def root_mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average'):
    """Root mean squared logarithmic error regression loss.

    Read more in the :ref:`User Guide <mean_squared_log_error>`.

    .. versionadded:: 1.4

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape             (n_outputs,), default='uniform_average'

        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.

        'raw_values' :
            Returns a full set of errors when the input is of multioutput
            format.

        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.

    Examples
    --------
    >>> from sklearn.metrics import root_mean_squared_log_error
    >>> y_true = [3, 5, 2.5, 7]
    >>> y_pred = [2.5, 5, 4, 8]
    >>> root_mean_squared_log_error(y_true, y_pred)
    0.199...
    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError('Root Mean Squared Logarithmic Error cannot be used when targets contain negative values.')
    return root_mean_squared_error(np.log1p(y_true), np.log1p(y_pred), sample_weight=sample_weight, multioutput=multioutput)