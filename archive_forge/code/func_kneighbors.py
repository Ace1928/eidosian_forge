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
def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
    """Find the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_queries, n_features),             or (n_queries, n_indexed) if metric == 'precomputed', default=None
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        n_neighbors : int, default=None
            Number of neighbors required for each sample. The default is the
            value passed to the constructor.

        return_distance : bool, default=True
            Whether or not to return the distances.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Array representing the lengths to points, only present if
            return_distance=True.

        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.

        Examples
        --------
        In the following example, we construct a NearestNeighbors
        class from an array representing our data set and ask who's
        the closest point to [1,1,1]

        >>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
        >>> from sklearn.neighbors import NearestNeighbors
        >>> neigh = NearestNeighbors(n_neighbors=1)
        >>> neigh.fit(samples)
        NearestNeighbors(n_neighbors=1)
        >>> print(neigh.kneighbors([[1., 1., 1.]]))
        (array([[0.5]]), array([[2]]))

        As you can see, it returns [[0.5]], and [[2]], which means that the
        element is at distance 0.5 and is the third element of samples
        (indexes start at 0). You can also query for multiple points:

        >>> X = [[0., 1., 0.], [1., 0., 1.]]
        >>> neigh.kneighbors(X, return_distance=False)
        array([[1],
               [2]]...)
        """
    check_is_fitted(self)
    if n_neighbors is None:
        n_neighbors = self.n_neighbors
    elif n_neighbors <= 0:
        raise ValueError('Expected n_neighbors > 0. Got %d' % n_neighbors)
    elif not isinstance(n_neighbors, numbers.Integral):
        raise TypeError('n_neighbors does not take %s value, enter integer value' % type(n_neighbors))
    query_is_train = X is None
    if query_is_train:
        X = self._fit_X
        n_neighbors += 1
    elif self.metric == 'precomputed':
        X = _check_precomputed(X)
    else:
        X = self._validate_data(X, accept_sparse='csr', reset=False, order='C')
    n_samples_fit = self.n_samples_fit_
    if n_neighbors > n_samples_fit:
        if query_is_train:
            n_neighbors -= 1
            inequality_str = 'n_neighbors < n_samples_fit'
        else:
            inequality_str = 'n_neighbors <= n_samples_fit'
        raise ValueError(f'Expected {inequality_str}, but n_neighbors = {n_neighbors}, n_samples_fit = {n_samples_fit}, n_samples = {X.shape[0]}')
    n_jobs = effective_n_jobs(self.n_jobs)
    chunked_results = None
    use_pairwise_distances_reductions = self._fit_method == 'brute' and ArgKmin.is_usable_for(X if X is not None else self._fit_X, self._fit_X, self.effective_metric_)
    if use_pairwise_distances_reductions:
        results = ArgKmin.compute(X=X, Y=self._fit_X, k=n_neighbors, metric=self.effective_metric_, metric_kwargs=self.effective_metric_params_, strategy='auto', return_distance=return_distance)
    elif self._fit_method == 'brute' and self.metric == 'precomputed' and issparse(X):
        results = _kneighbors_from_graph(X, n_neighbors=n_neighbors, return_distance=return_distance)
    elif self._fit_method == 'brute':
        reduce_func = partial(self._kneighbors_reduce_func, n_neighbors=n_neighbors, return_distance=return_distance)
        if self.effective_metric_ == 'euclidean':
            kwds = {'squared': True}
        else:
            kwds = self.effective_metric_params_
        chunked_results = list(pairwise_distances_chunked(X, self._fit_X, reduce_func=reduce_func, metric=self.effective_metric_, n_jobs=n_jobs, **kwds))
    elif self._fit_method in ['ball_tree', 'kd_tree']:
        if issparse(X):
            raise ValueError("%s does not work with sparse matrices. Densify the data, or set algorithm='brute'" % self._fit_method)
        chunked_results = Parallel(n_jobs, prefer='threads')((delayed(_tree_query_parallel_helper)(self._tree, X[s], n_neighbors, return_distance) for s in gen_even_slices(X.shape[0], n_jobs)))
    else:
        raise ValueError('internal: _fit_method not recognized')
    if chunked_results is not None:
        if return_distance:
            neigh_dist, neigh_ind = zip(*chunked_results)
            results = (np.vstack(neigh_dist), np.vstack(neigh_ind))
        else:
            results = np.vstack(chunked_results)
    if not query_is_train:
        return results
    else:
        if return_distance:
            neigh_dist, neigh_ind = results
        else:
            neigh_ind = results
        n_queries, _ = X.shape
        sample_range = np.arange(n_queries)[:, None]
        sample_mask = neigh_ind != sample_range
        dup_gr_nbrs = np.all(sample_mask, axis=1)
        sample_mask[:, 0][dup_gr_nbrs] = False
        neigh_ind = np.reshape(neigh_ind[sample_mask], (n_queries, n_neighbors - 1))
        if return_distance:
            neigh_dist = np.reshape(neigh_dist[sample_mask], (n_queries, n_neighbors - 1))
            return (neigh_dist, neigh_ind)
        return neigh_ind