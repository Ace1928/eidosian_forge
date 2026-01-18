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
@validate_params({'graph': ['sparse matrix'], 'copy': ['boolean'], 'warn_when_not_sorted': ['boolean']}, prefer_skip_nested_validation=True)
def sort_graph_by_row_values(graph, copy=False, warn_when_not_sorted=True):
    """Sort a sparse graph such that each row is stored with increasing values.

    .. versionadded:: 1.2

    Parameters
    ----------
    graph : sparse matrix of shape (n_samples, n_samples)
        Distance matrix to other samples, where only non-zero elements are
        considered neighbors. Matrix is converted to CSR format if not already.

    copy : bool, default=False
        If True, the graph is copied before sorting. If False, the sorting is
        performed inplace. If the graph is not of CSR format, `copy` must be
        True to allow the conversion to CSR format, otherwise an error is
        raised.

    warn_when_not_sorted : bool, default=True
        If True, a :class:`~sklearn.exceptions.EfficiencyWarning` is raised
        when the input graph is not sorted by row values.

    Returns
    -------
    graph : sparse matrix of shape (n_samples, n_samples)
        Distance matrix to other samples, where only non-zero elements are
        considered neighbors. Matrix is in CSR format.

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from sklearn.neighbors import sort_graph_by_row_values
    >>> X = csr_matrix(
    ...     [[0., 3., 1.],
    ...      [3., 0., 2.],
    ...      [1., 2., 0.]])
    >>> X.data
    array([3., 1., 3., 2., 1., 2.])
    >>> X_ = sort_graph_by_row_values(X)
    >>> X_.data
    array([1., 3., 2., 3., 1., 2.])
    """
    if graph.format == 'csr' and _is_sorted_by_data(graph):
        return graph
    if warn_when_not_sorted:
        warnings.warn('Precomputed sparse input was not sorted by row values. Use the function sklearn.neighbors.sort_graph_by_row_values to sort the input by row values, with warn_when_not_sorted=False to remove this warning.', EfficiencyWarning)
    if graph.format not in ('csr', 'csc', 'coo', 'lil'):
        raise TypeError(f'Sparse matrix in {graph.format!r} format is not supported due to its handling of explicit zeros')
    elif graph.format != 'csr':
        if not copy:
            raise ValueError('The input graph is not in CSR format. Use copy=True to allow the conversion to CSR format.')
        graph = graph.asformat('csr')
    elif copy:
        graph = graph.copy()
    row_nnz = np.diff(graph.indptr)
    if row_nnz.max() == row_nnz.min():
        n_samples = graph.shape[0]
        distances = graph.data.reshape(n_samples, -1)
        order = np.argsort(distances, kind='mergesort')
        order += np.arange(n_samples)[:, None] * row_nnz[0]
        order = order.ravel()
        graph.data = graph.data[order]
        graph.indices = graph.indices[order]
    else:
        for start, stop in zip(graph.indptr, graph.indptr[1:]):
            order = np.argsort(graph.data[start:stop], kind='mergesort')
            graph.data[start:stop] = graph.data[start:stop][order]
            graph.indices[start:stop] = graph.indices[start:stop][order]
    return graph