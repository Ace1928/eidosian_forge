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
@cupy._util.memoize(for_each_device=True)
def cupy_multiply_by_csr_step2():
    return cupy.ElementwiseKernel('T C_DATA, I C_INDICES, raw I FLAGS', 'raw D D_DATA, raw I D_INDICES', '\n        int j = FLAGS[i];\n        if (j < FLAGS[i+1]) {\n            D_DATA[j] = (D)(C_DATA);\n            D_INDICES[j] = C_INDICES;\n        }\n        ', 'cupyx_scipy_sparse_csr_multiply_by_csr_step2')