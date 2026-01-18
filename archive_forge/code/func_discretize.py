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
def discretize(vectors, *, copy=True, max_svd_restarts=30, n_iter_max=20, random_state=None):
    """Search for a partition matrix which is closest to the eigenvector embedding.

    This implementation was proposed in [1]_.

    Parameters
    ----------
    vectors : array-like of shape (n_samples, n_clusters)
        The embedding space of the samples.

    copy : bool, default=True
        Whether to copy vectors, or perform in-place normalization.

    max_svd_restarts : int, default=30
        Maximum number of attempts to restart SVD if convergence fails

    n_iter_max : int, default=30
        Maximum number of iterations to attempt in rotation and partition
        matrix search if machine precision convergence is not reached

    random_state : int, RandomState instance, default=None
        Determines random number generation for rotation matrix initialization.
        Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    labels : array of integers, shape: n_samples
        The labels of the clusters.

    References
    ----------

    .. [1] `Multiclass spectral clustering, 2003
           Stella X. Yu, Jianbo Shi
           <https://people.eecs.berkeley.edu/~jordan/courses/281B-spring04/readings/yu-shi.pdf>`_

    Notes
    -----

    The eigenvector embedding is used to iteratively search for the
    closest discrete partition.  First, the eigenvector embedding is
    normalized to the space of partition matrices. An optimal discrete
    partition matrix closest to this normalized embedding multiplied by
    an initial rotation is calculated.  Fixing this discrete partition
    matrix, an optimal rotation matrix is calculated.  These two
    calculations are performed until convergence.  The discrete partition
    matrix is returned as the clustering solution.  Used in spectral
    clustering, this method tends to be faster and more robust to random
    initialization than k-means.

    """
    random_state = check_random_state(random_state)
    vectors = as_float_array(vectors, copy=copy)
    eps = np.finfo(float).eps
    n_samples, n_components = vectors.shape
    norm_ones = np.sqrt(n_samples)
    for i in range(vectors.shape[1]):
        vectors[:, i] = vectors[:, i] / np.linalg.norm(vectors[:, i]) * norm_ones
        if vectors[0, i] != 0:
            vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])
    vectors = vectors / np.sqrt((vectors ** 2).sum(axis=1))[:, np.newaxis]
    svd_restarts = 0
    has_converged = False
    while svd_restarts < max_svd_restarts and (not has_converged):
        rotation = np.zeros((n_components, n_components))
        rotation[:, 0] = vectors[random_state.randint(n_samples), :].T
        c = np.zeros(n_samples)
        for j in range(1, n_components):
            c += np.abs(np.dot(vectors, rotation[:, j - 1]))
            rotation[:, j] = vectors[c.argmin(), :].T
        last_objective_value = 0.0
        n_iter = 0
        while not has_converged:
            n_iter += 1
            t_discrete = np.dot(vectors, rotation)
            labels = t_discrete.argmax(axis=1)
            vectors_discrete = csc_matrix((np.ones(len(labels)), (np.arange(0, n_samples), labels)), shape=(n_samples, n_components))
            t_svd = vectors_discrete.T * vectors
            try:
                U, S, Vh = np.linalg.svd(t_svd)
            except LinAlgError:
                svd_restarts += 1
                print('SVD did not converge, randomizing and trying again')
                break
            ncut_value = 2.0 * (n_samples - S.sum())
            if abs(ncut_value - last_objective_value) < eps or n_iter > n_iter_max:
                has_converged = True
            else:
                last_objective_value = ncut_value
                rotation = np.dot(Vh.T, U.T)
    if not has_converged:
        raise LinAlgError('SVD did not converge')
    return labels