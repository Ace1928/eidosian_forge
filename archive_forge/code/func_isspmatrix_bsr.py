from warnings import warn
import numpy as np
from ._matrix import spmatrix
from ._data import _data_matrix, _minmax_mixin
from ._compressed import _cs_matrix
from ._base import issparse, _formats, _spbase, sparray
from ._sputils import (isshape, getdtype, getdata, to_native, upcast,
from . import _sparsetools
from ._sparsetools import (bsr_matvec, bsr_matvecs, csr_matmat_maxnnz,
def isspmatrix_bsr(x):
    """Is `x` of a bsr_matrix type?

    Parameters
    ----------
    x
        object to check for being a bsr matrix

    Returns
    -------
    bool
        True if `x` is a bsr matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import bsr_array, bsr_matrix, csr_matrix, isspmatrix_bsr
    >>> isspmatrix_bsr(bsr_matrix([[5]]))
    True
    >>> isspmatrix_bsr(bsr_array([[5]]))
    False
    >>> isspmatrix_bsr(csr_matrix([[5]]))
    False
    """
    return isinstance(x, bsr_matrix)