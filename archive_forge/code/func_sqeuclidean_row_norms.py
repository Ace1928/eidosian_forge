from abc import abstractmethod
from typing import List
import numpy as np
from scipy.sparse import issparse
from ... import get_config
from .._dist_metrics import (
from ._argkmin import (
from ._argkmin_classmode import (
from ._base import _sqeuclidean_row_norms32, _sqeuclidean_row_norms64
from ._radius_neighbors import (
from ._radius_neighbors_classmode import (
def sqeuclidean_row_norms(X, num_threads):
    """Compute the squared euclidean norm of the rows of X in parallel.

    Parameters
    ----------
    X : ndarray or CSR matrix of shape (n_samples, n_features)
        Input data. Must be c-contiguous.

    num_threads : int
        The number of OpenMP threads to use.

    Returns
    -------
    sqeuclidean_row_norms : ndarray of shape (n_samples,)
        Arrays containing the squared euclidean norm of each row of X.
    """
    if X.dtype == np.float64:
        return np.asarray(_sqeuclidean_row_norms64(X, num_threads))
    if X.dtype == np.float32:
        return np.asarray(_sqeuclidean_row_norms32(X, num_threads))
    raise ValueError(f'Only float64 or float32 datasets are supported at this time, got: X.dtype={X.dtype}.')