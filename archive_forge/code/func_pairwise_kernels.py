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
@validate_params({'X': ['array-like', 'sparse matrix'], 'Y': ['array-like', 'sparse matrix', None], 'metric': [StrOptions(set(PAIRWISE_KERNEL_FUNCTIONS) | {'precomputed'}), callable], 'filter_params': ['boolean'], 'n_jobs': [Integral, None]}, prefer_skip_nested_validation=True)
def pairwise_kernels(X, Y=None, metric='linear', *, filter_params=False, n_jobs=None, **kwds):
    """Compute the kernel between arrays X and optional array Y.

    This method takes either a vector array or a kernel matrix, and returns
    a kernel matrix. If the input is a vector array, the kernels are
    computed. If the input is a kernel matrix, it is returned instead.

    This method provides a safe way to take a kernel matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.

    If Y is given (default is None), then the returned matrix is the pairwise
    kernel between the arrays from both X and Y.

    Valid values for metric are:
        ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf',
        'laplacian', 'sigmoid', 'cosine']

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix}  of shape (n_samples_X, n_samples_X) or             (n_samples_X, n_features)
        Array of pairwise kernels between samples, or a feature array.
        The shape of the array should be (n_samples_X, n_samples_X) if
        metric == "precomputed" and (n_samples_X, n_features) otherwise.

    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features), default=None
        A second feature array only if X has shape (n_samples_X, n_features).

    metric : str or callable, default="linear"
        The metric to use when calculating kernel between instances in a
        feature array. If metric is a string, it must be one of the metrics
        in ``pairwise.PAIRWISE_KERNEL_FUNCTIONS``.
        If metric is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two rows from X as input and return the corresponding
        kernel value as a single number. This means that callables from
        :mod:`sklearn.metrics.pairwise` are not allowed, as they operate on
        matrices, not single samples. Use the string identifying the kernel
        instead.

    filter_params : bool, default=False
        Whether to filter invalid parameters or not.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the kernel function.

    Returns
    -------
    K : ndarray of shape (n_samples_X, n_samples_X) or (n_samples_X, n_samples_Y)
        A kernel matrix K such that K_{i, j} is the kernel between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then K_{i, j} is the kernel between the ith array
        from X and the jth array from Y.

    Notes
    -----
    If metric is 'precomputed', Y is ignored and X is returned.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import pairwise_kernels
    >>> X = [[0, 0, 0], [1, 1, 1]]
    >>> Y = [[1, 0, 0], [1, 1, 0]]
    >>> pairwise_kernels(X, Y, metric='linear')
    array([[0., 0.],
           [1., 2.]])
    """
    from ..gaussian_process.kernels import Kernel as GPKernel
    if metric == 'precomputed':
        X, _ = check_pairwise_arrays(X, Y, precomputed=True)
        return X
    elif isinstance(metric, GPKernel):
        func = metric.__call__
    elif metric in PAIRWISE_KERNEL_FUNCTIONS:
        if filter_params:
            kwds = {k: kwds[k] for k in kwds if k in KERNEL_PARAMS[metric]}
        func = PAIRWISE_KERNEL_FUNCTIONS[metric]
    elif callable(metric):
        func = partial(_pairwise_callable, metric=metric, **kwds)
    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)