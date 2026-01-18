import itertools
import warnings
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy.sparse import csr_matrix, issparse
from scipy.spatial import distance
from .. import config_context
from ..exceptions import DataConversionWarning
from ..preprocessing import normalize
from ..utils import (
from ..utils._mask import _get_mask
from ..utils._param_validation import (
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.fixes import parse_version, sp_base_version
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _num_samples, check_non_negative
from ._pairwise_distances_reduction import ArgKmin
from ._pairwise_fast import _chi2_kernel_fast, _sparse_manhattan
@validate_params({'X': ['array-like'], 'Y': ['array-like', None]}, prefer_skip_nested_validation=True)
def additive_chi2_kernel(X, Y=None):
    """Compute the additive chi-squared kernel between observations in X and Y.

    The chi-squared kernel is computed between each pair of rows in X and Y.  X
    and Y have to be non-negative. This kernel is most commonly applied to
    histograms.

    The chi-squared kernel is given by::

        k(x, y) = -Sum [(x - y)^2 / (x + y)]

    It can be interpreted as a weighted difference per entry.

    Read more in the :ref:`User Guide <chi2_kernel>`.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        A feature array.

    Y : array-like of shape (n_samples_Y, n_features), default=None
        An optional second feature array. If `None`, uses `Y=X`.

    Returns
    -------
    kernel : ndarray of shape (n_samples_X, n_samples_Y)
        The kernel matrix.

    See Also
    --------
    chi2_kernel : The exponentiated version of the kernel, which is usually
        preferable.
    sklearn.kernel_approximation.AdditiveChi2Sampler : A Fourier approximation
        to this kernel.

    Notes
    -----
    As the negative of a distance, this kernel is only conditionally positive
    definite.

    References
    ----------
    * Zhang, J. and Marszalek, M. and Lazebnik, S. and Schmid, C.
      Local features and kernels for classification of texture and object
      categories: A comprehensive study
      International Journal of Computer Vision 2007
      https://hal.archives-ouvertes.fr/hal-00171412/document

    Examples
    --------
    >>> from sklearn.metrics.pairwise import additive_chi2_kernel
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> additive_chi2_kernel(X, Y)
    array([[-1., -2.],
           [-2., -1.]])
    """
    X, Y = check_pairwise_arrays(X, Y, accept_sparse=False)
    if (X < 0).any():
        raise ValueError('X contains negative values.')
    if Y is not X and (Y < 0).any():
        raise ValueError('Y contains negative values.')
    result = np.zeros((X.shape[0], Y.shape[0]), dtype=X.dtype)
    _chi2_kernel_fast(X, Y, result)
    return result