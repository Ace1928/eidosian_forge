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
def multiply_by_scalar(sp, a):
    data = sp.data * a
    indices = sp.indices.copy()
    indptr = sp.indptr.copy()
    return csr_matrix((data, indices, indptr), shape=sp.shape)