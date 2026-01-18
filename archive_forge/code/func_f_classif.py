import warnings
from numbers import Integral, Real
import numpy as np
from scipy import special, stats
from scipy.sparse import issparse
from ..base import BaseEstimator, _fit_context
from ..preprocessing import LabelBinarizer
from ..utils import as_float_array, check_array, check_X_y, safe_mask, safe_sqr
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.validation import check_is_fitted
from ._base import SelectorMixin
@validate_params({'X': ['array-like', 'sparse matrix'], 'y': ['array-like']}, prefer_skip_nested_validation=True)
def f_classif(X, y):
    """Compute the ANOVA F-value for the provided sample.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The set of regressors that will be tested sequentially.

    y : array-like of shape (n_samples,)
        The target vector.

    Returns
    -------
    f_statistic : ndarray of shape (n_features,)
        F-statistic for each feature.

    p_values : ndarray of shape (n_features,)
        P-values associated with the F-statistic.

    See Also
    --------
    chi2 : Chi-squared stats of non-negative features for classification tasks.
    f_regression : F-value between label/feature for regression tasks.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.feature_selection import f_classif
    >>> X, y = make_classification(
    ...     n_samples=100, n_features=10, n_informative=2, n_clusters_per_class=1,
    ...     shuffle=False, random_state=42
    ... )
    >>> f_statistic, p_values = f_classif(X, y)
    >>> f_statistic
    array([2.2...e+02, 7.0...e-01, 1.6...e+00, 9.3...e-01,
           5.4...e+00, 3.2...e-01, 4.7...e-02, 5.7...e-01,
           7.5...e-01, 8.9...e-02])
    >>> p_values
    array([7.1...e-27, 4.0...e-01, 1.9...e-01, 3.3...e-01,
           2.2...e-02, 5.7...e-01, 8.2...e-01, 4.5...e-01,
           3.8...e-01, 7.6...e-01])
    """
    X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'])
    args = [X[safe_mask(X, y == k)] for k in np.unique(y)]
    return f_oneway(*args)