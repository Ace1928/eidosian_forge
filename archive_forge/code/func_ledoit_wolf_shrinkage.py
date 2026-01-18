import warnings
from numbers import Integral, Real
import numpy as np
from ..base import _fit_context
from ..utils import check_array
from ..utils._param_validation import Interval, validate_params
from . import EmpiricalCovariance, empirical_covariance
@validate_params({'X': ['array-like'], 'assume_centered': ['boolean'], 'block_size': [Interval(Integral, 1, None, closed='left')]}, prefer_skip_nested_validation=True)
def ledoit_wolf_shrinkage(X, assume_centered=False, block_size=1000):
    """Estimate the shrunk Ledoit-Wolf covariance matrix.

    Read more in the :ref:`User Guide <shrunk_covariance>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data from which to compute the Ledoit-Wolf shrunk covariance shrinkage.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data will be centered before computation.

    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split.

    Returns
    -------
    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.

    Notes
    -----
    The regularized (shrunk) covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.covariance import ledoit_wolf_shrinkage
    >>> real_cov = np.array([[.4, .2], [.2, .8]])
    >>> rng = np.random.RandomState(0)
    >>> X = rng.multivariate_normal(mean=[0, 0], cov=real_cov, size=50)
    >>> shrinkage_coefficient = ledoit_wolf_shrinkage(X)
    >>> shrinkage_coefficient
    0.23...
    """
    X = check_array(X)
    if len(X.shape) == 2 and X.shape[1] == 1:
        return 0.0
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))
    if X.shape[0] == 1:
        warnings.warn('Only one sample available. You may want to reshape your data array')
    n_samples, n_features = X.shape
    if not assume_centered:
        X = X - X.mean(0)
    n_splits = int(n_features / block_size)
    X2 = X ** 2
    emp_cov_trace = np.sum(X2, axis=0) / n_samples
    mu = np.sum(emp_cov_trace) / n_features
    beta_ = 0.0
    delta_ = 0.0
    for i in range(n_splits):
        for j in range(n_splits):
            rows = slice(block_size * i, block_size * (i + 1))
            cols = slice(block_size * j, block_size * (j + 1))
            beta_ += np.sum(np.dot(X2.T[rows], X2[:, cols]))
            delta_ += np.sum(np.dot(X.T[rows], X[:, cols]) ** 2)
        rows = slice(block_size * i, block_size * (i + 1))
        beta_ += np.sum(np.dot(X2.T[rows], X2[:, block_size * n_splits:]))
        delta_ += np.sum(np.dot(X.T[rows], X[:, block_size * n_splits:]) ** 2)
    for j in range(n_splits):
        cols = slice(block_size * j, block_size * (j + 1))
        beta_ += np.sum(np.dot(X2.T[block_size * n_splits:], X2[:, cols]))
        delta_ += np.sum(np.dot(X.T[block_size * n_splits:], X[:, cols]) ** 2)
    delta_ += np.sum(np.dot(X.T[block_size * n_splits:], X[:, block_size * n_splits:]) ** 2)
    delta_ /= n_samples ** 2
    beta_ += np.sum(np.dot(X2.T[block_size * n_splits:], X2[:, block_size * n_splits:]))
    beta = 1.0 / (n_features * n_samples) * (beta_ / n_samples - delta_)
    delta = delta_ - 2.0 * mu * emp_cov_trace.sum() + n_features * mu ** 2
    delta /= n_features
    beta = min(beta, delta)
    shrinkage = 0 if beta == 0 else beta / delta
    return shrinkage