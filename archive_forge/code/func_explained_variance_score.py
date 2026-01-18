import warnings
from numbers import Real
import numpy as np
from scipy.special import xlogy
from ..exceptions import UndefinedMetricWarning
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.stats import _weighted_percentile
from ..utils.validation import (
@validate_params({'y_true': ['array-like'], 'y_pred': ['array-like'], 'sample_weight': ['array-like', None], 'multioutput': [StrOptions({'raw_values', 'uniform_average', 'variance_weighted'}), 'array-like'], 'force_finite': ['boolean']}, prefer_skip_nested_validation=True)
def explained_variance_score(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', force_finite=True):
    """Explained variance regression score function.

    Best possible score is 1.0, lower values are worse.

    In the particular case when ``y_true`` is constant, the explained variance
    score is not finite: it is either ``NaN`` (perfect predictions) or
    ``-Inf`` (imperfect predictions). To prevent such non-finite numbers to
    pollute higher-level experiments such as a grid search cross-validation,
    by default these cases are replaced with 1.0 (perfect predictions) or 0.0
    (imperfect predictions) respectively. If ``force_finite``
    is set to ``False``, this score falls back on the original :math:`R^2`
    definition.

    .. note::
       The Explained Variance score is similar to the
       :func:`R^2 score <r2_score>`, with the notable difference that it
       does not account for systematic offsets in the prediction. Most often
       the :func:`R^2 score <r2_score>` should be preferred.

    Read more in the :ref:`User Guide <explained_variance_score>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    multioutput : {'raw_values', 'uniform_average', 'variance_weighted'} or             array-like of shape (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output scores.
        Array-like value defines weights used to average scores.

        'raw_values' :
            Returns a full set of scores in case of multioutput input.

        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.

        'variance_weighted' :
            Scores of all outputs are averaged, weighted by the variances
            of each individual output.

    force_finite : bool, default=True
        Flag indicating if ``NaN`` and ``-Inf`` scores resulting from constant
        data should be replaced with real numbers (``1.0`` if prediction is
        perfect, ``0.0`` otherwise). Default is ``True``, a convenient setting
        for hyperparameters' search procedures (e.g. grid search
        cross-validation).

        .. versionadded:: 1.1

    Returns
    -------
    score : float or ndarray of floats
        The explained variance or ndarray if 'multioutput' is 'raw_values'.

    See Also
    --------
    r2_score :
        Similar metric, but accounting for systematic offsets in
        prediction.

    Notes
    -----
    This is not a symmetric function.

    Examples
    --------
    >>> from sklearn.metrics import explained_variance_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> explained_variance_score(y_true, y_pred)
    0.957...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> explained_variance_score(y_true, y_pred, multioutput='uniform_average')
    0.983...
    >>> y_true = [-2, -2, -2]
    >>> y_pred = [-2, -2, -2]
    >>> explained_variance_score(y_true, y_pred)
    1.0
    >>> explained_variance_score(y_true, y_pred, force_finite=False)
    nan
    >>> y_true = [-2, -2, -2]
    >>> y_pred = [-2, -2, -2 + 1e-8]
    >>> explained_variance_score(y_true, y_pred)
    0.0
    >>> explained_variance_score(y_true, y_pred, force_finite=False)
    -inf
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    y_diff_avg = np.average(y_true - y_pred, weights=sample_weight, axis=0)
    numerator = np.average((y_true - y_pred - y_diff_avg) ** 2, weights=sample_weight, axis=0)
    y_true_avg = np.average(y_true, weights=sample_weight, axis=0)
    denominator = np.average((y_true - y_true_avg) ** 2, weights=sample_weight, axis=0)
    return _assemble_r2_explained_variance(numerator=numerator, denominator=denominator, n_outputs=y_true.shape[1], multioutput=multioutput, force_finite=force_finite)