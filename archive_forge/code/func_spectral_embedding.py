import warnings
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh, lobpcg
from ..base import BaseEstimator, _fit_context
from ..metrics.pairwise import rbf_kernel
from ..neighbors import NearestNeighbors, kneighbors_graph
from ..utils import (
from ..utils._arpack import _init_arpack_v0
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import _deterministic_vector_sign_flip
from ..utils.fixes import laplacian as csgraph_laplacian
from ..utils.fixes import parse_version, sp_version
def spectral_embedding(adjacency, *, n_components=8, eigen_solver=None, random_state=None, eigen_tol='auto', norm_laplacian=True, drop_first=True):
    """Project the sample on the first eigenvectors of the graph Laplacian.

    The adjacency matrix is used to compute a normalized graph Laplacian
    whose spectrum (especially the eigenvectors associated to the
    smallest eigenvalues) has an interpretation in terms of minimal
    number of cuts necessary to split the graph into comparably sized
    components.

    This embedding can also 'work' even if the ``adjacency`` variable is
    not strictly the adjacency matrix of a graph but more generally
    an affinity or similarity matrix between samples (for instance the
    heat kernel of a euclidean distance matrix or a k-NN matrix).

    However care must taken to always make the affinity matrix symmetric
    so that the eigenvector decomposition works as expected.

    Note : Laplacian Eigenmaps is the actual algorithm implemented here.

    Read more in the :ref:`User Guide <spectral_embedding>`.

    Parameters
    ----------
    adjacency : {array-like, sparse graph} of shape (n_samples, n_samples)
        The adjacency matrix of the graph to embed.

    n_components : int, default=8
        The dimension of the projection subspace.

    eigen_solver : {'arpack', 'lobpcg', 'amg'}, default=None
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities. If None, then ``'arpack'`` is
        used.

    random_state : int, RandomState instance or None, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigen vectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls (See
        :term:`Glossary <random_state>`).

        .. note::
            When using `eigen_solver == 'amg'`,
            it is necessary to also fix the global numpy seed with
            `np.random.seed(int)` to get deterministic results. See
            https://github.com/pyamg/pyamg/issues/139 for further
            information.

    eigen_tol : float, default="auto"
        Stopping criterion for eigendecomposition of the Laplacian matrix.
        If `eigen_tol="auto"` then the passed tolerance will depend on the
        `eigen_solver`:

        - If `eigen_solver="arpack"`, then `eigen_tol=0.0`;
        - If `eigen_solver="lobpcg"` or `eigen_solver="amg"`, then
          `eigen_tol=None` which configures the underlying `lobpcg` solver to
          automatically resolve the value according to their heuristics. See,
          :func:`scipy.sparse.linalg.lobpcg` for details.

        Note that when using `eigen_solver="amg"` values of `tol<1e-5` may lead
        to convergence issues and should be avoided.

        .. versionadded:: 1.2
           Added 'auto' option.

    norm_laplacian : bool, default=True
        If True, then compute symmetric normalized Laplacian.

    drop_first : bool, default=True
        Whether to drop the first eigenvector. For spectral embedding, this
        should be True as the first eigenvector should be constant vector for
        connected graph, but for spectral clustering, this should be kept as
        False to retain the first eigenvector.

    Returns
    -------
    embedding : ndarray of shape (n_samples, n_components)
        The reduced samples.

    Notes
    -----
    Spectral Embedding (Laplacian Eigenmaps) is most useful when the graph
    has one connected component. If there graph has many components, the first
    few eigenvectors will simply uncover the connected components of the graph.

    References
    ----------
    * https://en.wikipedia.org/wiki/LOBPCG

    * :doi:`"Toward the Optimal Preconditioned Eigensolver: Locally Optimal
      Block Preconditioned Conjugate Gradient Method",
      Andrew V. Knyazev
      <10.1137/S1064827500366124>`

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.neighbors import kneighbors_graph
    >>> from sklearn.manifold import spectral_embedding
    >>> X, _ = load_digits(return_X_y=True)
    >>> X = X[:100]
    >>> affinity_matrix = kneighbors_graph(
    ...     X, n_neighbors=int(X.shape[0] / 10), include_self=True
    ... )
    >>> # make the matrix symmetric
    >>> affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)
    >>> embedding = spectral_embedding(affinity_matrix, n_components=2, random_state=42)
    >>> embedding.shape
    (100, 2)
    """
    adjacency = check_symmetric(adjacency)
    if eigen_solver == 'amg':
        try:
            from pyamg import smoothed_aggregation_solver
        except ImportError as e:
            raise ValueError("The eigen_solver was set to 'amg', but pyamg is not available.") from e
    if eigen_solver is None:
        eigen_solver = 'arpack'
    elif eigen_solver not in ('arpack', 'lobpcg', 'amg'):
        raise ValueError("Unknown value for eigen_solver: '%s'.Should be 'amg', 'arpack', or 'lobpcg'" % eigen_solver)
    random_state = check_random_state(random_state)
    n_nodes = adjacency.shape[0]
    if drop_first:
        n_components = n_components + 1
    if not _graph_is_connected(adjacency):
        warnings.warn('Graph is not fully connected, spectral embedding may not work as expected.')
    laplacian, dd = csgraph_laplacian(adjacency, normed=norm_laplacian, return_diag=True)
    if eigen_solver == 'arpack' or (eigen_solver != 'lobpcg' and (not sparse.issparse(laplacian) or n_nodes < 5 * n_components)):
        laplacian = _set_diag(laplacian, 1, norm_laplacian)
        try:
            tol = 0 if eigen_tol == 'auto' else eigen_tol
            laplacian *= -1
            v0 = _init_arpack_v0(laplacian.shape[0], random_state)
            laplacian = check_array(laplacian, accept_sparse='csr', accept_large_sparse=False)
            _, diffusion_map = eigsh(laplacian, k=n_components, sigma=1.0, which='LM', tol=tol, v0=v0)
            embedding = diffusion_map.T[n_components::-1]
            if norm_laplacian:
                embedding = embedding / dd
        except RuntimeError:
            eigen_solver = 'lobpcg'
            laplacian *= -1
    elif eigen_solver == 'amg':
        if not sparse.issparse(laplacian):
            warnings.warn('AMG works better for sparse matrices')
        laplacian = check_array(laplacian, dtype=[np.float64, np.float32], accept_sparse=True)
        laplacian = _set_diag(laplacian, 1, norm_laplacian)
        diag_shift = 1e-05 * sparse.eye(laplacian.shape[0])
        laplacian += diag_shift
        if hasattr(sparse, 'csr_array') and isinstance(laplacian, sparse.csr_array):
            laplacian = sparse.csr_matrix(laplacian)
        ml = smoothed_aggregation_solver(check_array(laplacian, accept_sparse='csr'))
        laplacian -= diag_shift
        M = ml.aspreconditioner()
        X = random_state.standard_normal(size=(laplacian.shape[0], n_components + 1))
        X[:, 0] = dd.ravel()
        X = X.astype(laplacian.dtype)
        tol = None if eigen_tol == 'auto' else eigen_tol
        _, diffusion_map = lobpcg(laplacian, X, M=M, tol=tol, largest=False)
        embedding = diffusion_map.T
        if norm_laplacian:
            embedding = embedding / dd
        if embedding.shape[0] == 1:
            raise ValueError
    if eigen_solver == 'lobpcg':
        laplacian = check_array(laplacian, dtype=[np.float64, np.float32], accept_sparse=True)
        if n_nodes < 5 * n_components + 1:
            if sparse.issparse(laplacian):
                laplacian = laplacian.toarray()
            _, diffusion_map = eigh(laplacian, check_finite=False)
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                embedding = embedding / dd
        else:
            laplacian = _set_diag(laplacian, 1, norm_laplacian)
            X = random_state.standard_normal(size=(laplacian.shape[0], n_components + 1))
            X[:, 0] = dd.ravel()
            X = X.astype(laplacian.dtype)
            tol = None if eigen_tol == 'auto' else eigen_tol
            _, diffusion_map = lobpcg(laplacian, X, tol=tol, largest=False, maxiter=2000)
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                embedding = embedding / dd
            if embedding.shape[0] == 1:
                raise ValueError
    embedding = _deterministic_vector_sign_flip(embedding)
    if drop_first:
        return embedding[1:n_components].T
    else:
        return embedding[:n_components].T