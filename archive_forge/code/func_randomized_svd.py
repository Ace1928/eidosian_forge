import warnings
from functools import partial
from numbers import Integral
import numpy as np
from scipy import linalg, sparse
from ..utils import deprecated
from ..utils._param_validation import Interval, StrOptions, validate_params
from . import check_random_state
from ._array_api import _is_numpy_namespace, device, get_namespace
from .sparsefuncs_fast import csr_row_norms
from .validation import check_array
@validate_params({'M': [np.ndarray, 'sparse matrix'], 'n_components': [Interval(Integral, 1, None, closed='left')], 'n_oversamples': [Interval(Integral, 0, None, closed='left')], 'n_iter': [Interval(Integral, 0, None, closed='left'), StrOptions({'auto'})], 'power_iteration_normalizer': [StrOptions({'auto', 'QR', 'LU', 'none'})], 'transpose': ['boolean', StrOptions({'auto'})], 'flip_sign': ['boolean'], 'random_state': ['random_state'], 'svd_lapack_driver': [StrOptions({'gesdd', 'gesvd'})]}, prefer_skip_nested_validation=True)
def randomized_svd(M, n_components, *, n_oversamples=10, n_iter='auto', power_iteration_normalizer='auto', transpose='auto', flip_sign=True, random_state=None, svd_lapack_driver='gesdd'):
    """Compute a truncated randomized SVD.

    This method solves the fixed-rank approximation problem described in [1]_
    (problem (1.5), p5).

    Parameters
    ----------
    M : {ndarray, sparse matrix}
        Matrix to decompose.

    n_components : int
        Number of singular values and vectors to extract.

    n_oversamples : int, default=10
        Additional number of random vectors to sample the range of `M` so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of `M` is `n_components + n_oversamples`. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values. Users might wish
        to increase this parameter up to `2*k - n_components` where k is the
        effective rank, for large matrices, noisy problems, matrices with
        slowly decaying spectrums, or to increase precision accuracy. See [1]_
        (pages 5, 23 and 26).

    n_iter : int or 'auto', default='auto'
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) in which case `n_iter` is set to 7.
        This improves precision with few components. Note that in general
        users should rather increase `n_oversamples` before increasing `n_iter`
        as the principle of the randomized method is to avoid usage of these
        more costly power iterations steps. When `n_components` is equal
        or greater to the effective matrix rank and the spectrum does not
        present a slow decay, `n_iter=0` or `1` should even work fine in theory
        (see [1]_ page 9).

        .. versionchanged:: 0.18

    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

        .. versionadded:: 0.18

    transpose : bool or 'auto', default='auto'
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.

        .. versionchanged:: 0.18

    flip_sign : bool, default=True
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.

    random_state : int, RandomState instance or None, default='warn'
        The seed of the pseudo random number generator to use when
        shuffling the data, i.e. getting the random vectors to initialize
        the algorithm. Pass an int for reproducible results across multiple
        function calls. See :term:`Glossary <random_state>`.

        .. versionchanged:: 1.2
            The default value changed from 0 to None.

    svd_lapack_driver : {"gesdd", "gesvd"}, default="gesdd"
        Whether to use the more efficient divide-and-conquer approach
        (`"gesdd"`) or more general rectangular approach (`"gesvd"`) to compute
        the SVD of the matrix B, which is the projection of M into a low
        dimensional subspace, as described in [1]_.

        .. versionadded:: 1.2

    Returns
    -------
    u : ndarray of shape (n_samples, n_components)
        Unitary matrix having left singular vectors with signs flipped as columns.
    s : ndarray of shape (n_components,)
        The singular values, sorted in non-increasing order.
    vh : ndarray of shape (n_components, n_features)
        Unitary matrix having right singular vectors with signs flipped as rows.

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision). To increase the precision it is recommended to
    increase `n_oversamples`, up to `2*k-n_components` where k is the
    effective rank. Usually, `n_components` is chosen to be greater than k
    so increasing `n_oversamples` up to `n_components` should be enough.

    References
    ----------
    .. [1] :arxiv:`"Finding structure with randomness:
      Stochastic algorithms for constructing approximate matrix decompositions"
      <0909.4061>`
      Halko, et al. (2009)

    .. [2] A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

    .. [3] An implementation of a randomized algorithm for principal component
      analysis A. Szlam et al. 2014

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.extmath import randomized_svd
    >>> a = np.array([[1, 2, 3, 5],
    ...               [3, 4, 5, 6],
    ...               [7, 8, 9, 10]])
    >>> U, s, Vh = randomized_svd(a, n_components=2, random_state=0)
    >>> U.shape, s.shape, Vh.shape
    ((3, 2), (2,), (2, 4))
    """
    if sparse.issparse(M) and M.format in ('lil', 'dok'):
        warnings.warn('Calculating SVD of a {} is expensive. csr_matrix is more efficient.'.format(type(M).__name__), sparse.SparseEfficiencyWarning)
    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape
    if n_iter == 'auto':
        n_iter = 7 if n_components < 0.1 * min(M.shape) else 4
    if transpose == 'auto':
        transpose = n_samples < n_features
    if transpose:
        M = M.T
    Q = randomized_range_finder(M, size=n_random, n_iter=n_iter, power_iteration_normalizer=power_iteration_normalizer, random_state=random_state)
    B = Q.T @ M
    xp, is_array_api_compliant = get_namespace(B)
    if is_array_api_compliant:
        Uhat, s, Vt = xp.linalg.svd(B, full_matrices=False)
    else:
        Uhat, s, Vt = linalg.svd(B, full_matrices=False, lapack_driver=svd_lapack_driver)
    del B
    U = Q @ Uhat
    if flip_sign:
        if not transpose:
            U, Vt = svd_flip(U, Vt)
        else:
            U, Vt = svd_flip(U, Vt, u_based_decision=False)
    if transpose:
        return (Vt[:n_components, :].T, s[:n_components], U[:, :n_components].T)
    else:
        return (U[:, :n_components], s[:n_components], Vt[:n_components, :])