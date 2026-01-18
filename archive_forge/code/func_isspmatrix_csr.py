import operator
import warnings
import numpy
import cupy
from cupy._core import _accelerator
from cupy.cuda import cub
from cupy.cuda import runtime
from cupyx import cusparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _compressed
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import SparseEfficiencyWarning
from cupyx.scipy.sparse import _util
def isspmatrix_csr(x):
    """Checks if a given matrix is of CSR format.

    Returns:
        bool: Returns if ``x`` is :class:`cupyx.scipy.sparse.csr_matrix`.

    """
    return isinstance(x, csr_matrix)