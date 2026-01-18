import itertools
import numbers
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy.sparse import csr_matrix, issparse
from ..base import BaseEstimator, MultiOutputMixin, is_classifier
from ..exceptions import DataConversionWarning, EfficiencyWarning
from ..metrics import DistanceMetric, pairwise_distances_chunked
from ..metrics._pairwise_distances_reduction import (
from ..metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from ..utils import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.fixes import parse_version, sp_base_version
from ..utils.multiclass import check_classification_targets
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted, check_non_negative
from ._ball_tree import BallTree
from ._kd_tree import KDTree
def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
    """Compute the (weighted) graph of k-Neighbors for points in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features),             or (n_queries, n_indexed) if metric == 'precomputed', default=None
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.
            For ``metric='precomputed'`` the shape should be
            (n_queries, n_indexed). Otherwise the shape should be
            (n_queries, n_features).

        n_neighbors : int, default=None
            Number of neighbors for each sample. The default is the value
            passed to the constructor.

        mode : {'connectivity', 'distance'}, default='connectivity'
            Type of returned matrix: 'connectivity' will return the
            connectivity matrix with ones and zeros, in 'distance' the
            edges are distances between points, type of distance
            depends on the selected metric parameter in
            NearestNeighbors class.

        Returns
        -------
        A : sparse-matrix of shape (n_queries, n_samples_fit)
            `n_samples_fit` is the number of samples in the fitted data.
            `A[i, j]` gives the weight of the edge connecting `i` to `j`.
            The matrix is of CSR format.

        See Also
        --------
        NearestNeighbors.radius_neighbors_graph : Compute the (weighted) graph
            of Neighbors for points in X.

        Examples
        --------
        >>> X = [[0], [3], [1]]
        >>> from sklearn.neighbors import NearestNeighbors
        >>> neigh = NearestNeighbors(n_neighbors=2)
        >>> neigh.fit(X)
        NearestNeighbors(n_neighbors=2)
        >>> A = neigh.kneighbors_graph(X)
        >>> A.toarray()
        array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 0., 1.]])
        """
    check_is_fitted(self)
    if n_neighbors is None:
        n_neighbors = self.n_neighbors
    if mode == 'connectivity':
        A_ind = self.kneighbors(X, n_neighbors, return_distance=False)
        n_queries = A_ind.shape[0]
        A_data = np.ones(n_queries * n_neighbors)
    elif mode == 'distance':
        A_data, A_ind = self.kneighbors(X, n_neighbors, return_distance=True)
        A_data = np.ravel(A_data)
    else:
        raise ValueError(f'Unsupported mode, must be one of "connectivity", or "distance" but got "{mode}" instead')
    n_queries = A_ind.shape[0]
    n_samples_fit = self.n_samples_fit_
    n_nonzero = n_queries * n_neighbors
    A_indptr = np.arange(0, n_nonzero + 1, n_neighbors)
    kneighbors_graph = csr_matrix((A_data, A_ind.ravel(), A_indptr), shape=(n_queries, n_samples_fit))
    return kneighbors_graph