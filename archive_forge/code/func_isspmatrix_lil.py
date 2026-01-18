from bisect import bisect_left
import numpy as np
from ._matrix import spmatrix
from ._base import _spbase, sparray, issparse
from ._index import IndexMixin, INT_TYPES, _broadcast_arrays
from ._sputils import (getdtype, isshape, isscalarlike, upcast_scalar,
from . import _csparsetools
def isspmatrix_lil(x):
    """Is `x` of lil_matrix type?

    Parameters
    ----------
    x
        object to check for being a lil matrix

    Returns
    -------
    bool
        True if `x` is a lil matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import lil_array, lil_matrix, coo_matrix, isspmatrix_lil
    >>> isspmatrix_lil(lil_matrix([[5]]))
    True
    >>> isspmatrix_lil(lil_array([[5]]))
    False
    >>> isspmatrix_lil(coo_matrix([[5]]))
    False
    """
    return isinstance(x, lil_matrix)