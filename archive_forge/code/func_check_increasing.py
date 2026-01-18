import math
import warnings
from numbers import Real
import numpy as np
from scipy import interpolate
from scipy.stats import spearmanr
from ._isotonic import _inplace_contiguous_isotonic_regression, _make_unique
from .base import BaseEstimator, RegressorMixin, TransformerMixin, _fit_context
from .utils import check_array, check_consistent_length
from .utils._param_validation import Interval, StrOptions, validate_params
from .utils.validation import _check_sample_weight, check_is_fitted
@validate_params({'x': ['array-like'], 'y': ['array-like']}, prefer_skip_nested_validation=True)
def check_increasing(x, y):
    """Determine whether y is monotonically correlated with x.

    y is found increasing or decreasing with respect to x based on a Spearman
    correlation test.

    Parameters
    ----------
    x : array-like of shape (n_samples,)
            Training data.

    y : array-like of shape (n_samples,)
        Training target.

    Returns
    -------
    increasing_bool : boolean
        Whether the relationship is increasing or decreasing.

    Notes
    -----
    The Spearman correlation coefficient is estimated from the data, and the
    sign of the resulting estimate is used as the result.

    In the event that the 95% confidence interval based on Fisher transform
    spans zero, a warning is raised.

    References
    ----------
    Fisher transformation. Wikipedia.
    https://en.wikipedia.org/wiki/Fisher_transformation

    Examples
    --------
    >>> from sklearn.isotonic import check_increasing
    >>> x, y = [1, 2, 3, 4, 5], [2, 4, 6, 8, 10]
    >>> check_increasing(x, y)
    True
    >>> y = [10, 8, 6, 4, 2]
    >>> check_increasing(x, y)
    False
    """
    rho, _ = spearmanr(x, y)
    increasing_bool = rho >= 0
    if rho not in [-1.0, 1.0] and len(x) > 3:
        F = 0.5 * math.log((1.0 + rho) / (1.0 - rho))
        F_se = 1 / math.sqrt(len(x) - 3)
        rho_0 = math.tanh(F - 1.96 * F_se)
        rho_1 = math.tanh(F + 1.96 * F_se)
        if np.sign(rho_0) != np.sign(rho_1):
            warnings.warn('Confidence interval of the Spearman correlation coefficient spans zero. Determination of ``increasing`` may be suspect.')
    return increasing_bool