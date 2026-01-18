from itertools import product
from numbers import Integral, Number, Real
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import sparse
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import check_array, check_random_state
from ..utils._param_validation import Hidden, Interval, RealNotInt, validate_params
@validate_params({'n_x': [Interval(Integral, left=1, right=None, closed='left')], 'n_y': [Interval(Integral, left=1, right=None, closed='left')], 'n_z': [Interval(Integral, left=1, right=None, closed='left')], 'mask': [None, np.ndarray], 'return_as': [type], 'dtype': 'no_validation'}, prefer_skip_nested_validation=True)
def grid_to_graph(n_x, n_y, n_z=1, *, mask=None, return_as=sparse.coo_matrix, dtype=int):
    """Graph of the pixel-to-pixel connections.

    Edges exist if 2 voxels are connected.

    Parameters
    ----------
    n_x : int
        Dimension in x axis.
    n_y : int
        Dimension in y axis.
    n_z : int, default=1
        Dimension in z axis.
    mask : ndarray of shape (n_x, n_y, n_z), dtype=bool, default=None
        An optional mask of the image, to consider only part of the
        pixels.
    return_as : np.ndarray or a sparse matrix class,             default=sparse.coo_matrix
        The class to use to build the returned adjacency matrix.
    dtype : dtype, default=int
        The data of the returned sparse matrix. By default it is int.

    Returns
    -------
    graph : np.ndarray or a sparse matrix class
        The computed adjacency matrix.

    Notes
    -----
    For scikit-learn versions 0.14.1 and prior, return_as=np.ndarray was
    handled by returning a dense np.matrix instance.  Going forward, np.ndarray
    returns an np.ndarray, as expected.

    For compatibility, user code relying on this method should wrap its
    calls in ``np.asarray`` to avoid type issues.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.feature_extraction.image import grid_to_graph
    >>> shape_img = (4, 4, 1)
    >>> mask = np.zeros(shape=shape_img, dtype=bool)
    >>> mask[[1, 2], [1, 2], :] = True
    >>> graph = grid_to_graph(*shape_img, mask=mask)
    >>> print(graph)
      (0, 0)    1
      (1, 1)    1
    """
    return _to_graph(n_x, n_y, n_z, mask=mask, return_as=return_as, dtype=dtype)