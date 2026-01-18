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
def cupy_dense2csr_step2():
    return cupy.ElementwiseKernel('int32 M, int32 N, T A, raw I INFO', 'raw I INDICES, raw T DATA', '\n        int row = i / N;\n        int col = i % N;\n        if (A != static_cast<T>(0)) {\n            int idx = INFO[i];\n            INDICES[idx] = col;\n            DATA[idx] = A;\n        }\n        ', 'cupyx_scipy_sparse_dense2csr_step2')