import warnings
from numbers import Integral, Real
import numpy as np
from scipy.linalg import LinAlgError, qr, svd
from scipy.sparse import csc_matrix
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..manifold import spectral_embedding
from ..metrics.pairwise import KERNEL_PARAMS, pairwise_kernels
from ..neighbors import NearestNeighbors, kneighbors_graph
from ..utils import as_float_array, check_random_state
from ..utils._param_validation import Interval, StrOptions, validate_params
from ._kmeans import k_means
@validate_params({'affinity': ['array-like', 'sparse matrix']}, prefer_skip_nested_validation=False)
def spectral_clustering(affinity, *, n_clusters=8, n_components=None, eigen_solver=None, random_state=None, n_init=10, eigen_tol='auto', assign_labels='kmeans', verbose=False):
    """Apply clustering to a projection of the normalized Laplacian.

    In practice Spectral Clustering is very useful when the structure of
    the individual clusters is highly non-convex or more generally when
    a measure of the center and spread of the cluster is not a suitable
    description of the complete cluster. For instance, when clusters are
    nested circles on the 2D plane.

    If affinity is the adjacency matrix of a graph, this method can be
    used to find normalized graph cuts [1]_, [2]_.

    Read more in the :ref:`User Guide <spectral_clustering>`.

    Parameters
    ----------
    affinity : {array-like, sparse matrix} of shape (n_samples, n_samples)
        The affinity matrix describing the relationship of the samples to
        embed. **Must be symmetric**.

        Possible examples:
          - adjacency matrix of a graph,
          - heat kernel of the pairwise distance matrix of the samples,
          - symmetric k-nearest neighbours connectivity matrix of the samples.

    n_clusters : int, default=None
        Number of clusters to extract.

    n_components : int, default=n_clusters
        Number of eigenvectors to use for the spectral embedding.

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition method. If None then ``'arpack'`` is used.
        See [4]_ for more details regarding ``'lobpcg'``.
        Eigensolver ``'amg'`` runs ``'lobpcg'`` with optional
        Algebraic MultiGrid preconditioning and requires pyamg to be installed.
        It can be faster on very large sparse problems [6]_ and [7]_.

    random_state : int, RandomState instance, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigenvectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls (See
        :term:`Glossary <random_state>`).

        .. note::
            When using `eigen_solver == 'amg'`,
            it is necessary to also fix the global numpy seed with
            `np.random.seed(int)` to get deterministic results. See
            https://github.com/pyamg/pyamg/issues/139 for further
            information.

    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of n_init
        consecutive runs in terms of inertia. Only used if
        ``assign_labels='kmeans'``.

    eigen_tol : float, default="auto"
        Stopping criterion for eigendecomposition of the Laplacian matrix.
        If `eigen_tol="auto"` then the passed tolerance will depend on the
        `eigen_solver`:

        - If `eigen_solver="arpack"`, then `eigen_tol=0.0`;
        - If `eigen_solver="lobpcg"` or `eigen_solver="amg"`, then
          `eigen_tol=None` which configures the underlying `lobpcg` solver to
          automatically resolve the value according to their heuristics. See,
          :func:`scipy.sparse.linalg.lobpcg` for details.

        Note that when using `eigen_solver="lobpcg"` or `eigen_solver="amg"`
        values of `tol<1e-5` may lead to convergence issues and should be
        avoided.

        .. versionadded:: 1.2
           Added 'auto' option.

    assign_labels : {'kmeans', 'discretize', 'cluster_qr'}, default='kmeans'
        The strategy to use to assign labels in the embedding
        space.  There are three ways to assign labels after the Laplacian
        embedding.  k-means can be applied and is a popular choice. But it can
        also be sensitive to initialization. Discretization is another
        approach which is less sensitive to random initialization [3]_.
        The cluster_qr method [5]_ directly extracts clusters from eigenvectors
        in spectral clustering. In contrast to k-means and discretization, cluster_qr
        has no tuning parameters and is not an iterative method, yet may outperform
        k-means and discretization in terms of both quality and speed.

        .. versionchanged:: 1.1
           Added new labeling method 'cluster_qr'.

    verbose : bool, default=False
        Verbosity mode.

        .. versionadded:: 0.24

    Returns
    -------
    labels : array of integers, shape: n_samples
        The labels of the clusters.

    Notes
    -----
    The graph should contain only one connected component, elsewhere
    the results make little sense.

    This algorithm solves the normalized cut for `k=2`: it is a
    normalized spectral clustering.

    References
    ----------

    .. [1] :doi:`Normalized cuts and image segmentation, 2000
           Jianbo Shi, Jitendra Malik
           <10.1109/34.868688>`

    .. [2] :doi:`A Tutorial on Spectral Clustering, 2007
           Ulrike von Luxburg
           <10.1007/s11222-007-9033-z>`

    .. [3] `Multiclass spectral clustering, 2003
           Stella X. Yu, Jianbo Shi
           <https://people.eecs.berkeley.edu/~jordan/courses/281B-spring04/readings/yu-shi.pdf>`_

    .. [4] :doi:`Toward the Optimal Preconditioned Eigensolver:
           Locally Optimal Block Preconditioned Conjugate Gradient Method, 2001
           A. V. Knyazev
           SIAM Journal on Scientific Computing 23, no. 2, pp. 517-541.
           <10.1137/S1064827500366124>`

    .. [5] :doi:`Simple, direct, and efficient multi-way spectral clustering, 2019
           Anil Damle, Victor Minden, Lexing Ying
           <10.1093/imaiai/iay008>`

    .. [6] :doi:`Multiscale Spectral Image Segmentation Multiscale preconditioning
           for computing eigenvalues of graph Laplacians in image segmentation, 2006
           Andrew Knyazev
           <10.13140/RG.2.2.35280.02565>`

    .. [7] :doi:`Preconditioned spectral clustering for stochastic block partition
           streaming graph challenge (Preliminary version at arXiv.)
           David Zhuzhunashvili, Andrew Knyazev
           <10.1109/HPEC.2017.8091045>`

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics.pairwise import pairwise_kernels
    >>> from sklearn.cluster import spectral_clustering
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> affinity = pairwise_kernels(X, metric='rbf')
    >>> spectral_clustering(
    ...     affinity=affinity, n_clusters=2, assign_labels="discretize", random_state=0
    ... )
    array([1, 1, 1, 0, 0, 0])
    """
    clusterer = SpectralClustering(n_clusters=n_clusters, n_components=n_components, eigen_solver=eigen_solver, random_state=random_state, n_init=n_init, affinity='precomputed', eigen_tol=eigen_tol, assign_labels=assign_labels, verbose=verbose).fit(affinity)
    return clusterer.labels_