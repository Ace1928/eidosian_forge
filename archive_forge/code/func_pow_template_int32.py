import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
def pow_template_int32(self, npdtype):
    nelem = 50
    A = np.linspace(0, 25, nelem).astype(npdtype)
    B = np.arange(nelem, dtype=np.int32)
    C = np.empty_like(A)
    arytype = numpy_support.from_dtype(npdtype)[::1]
    cfunc = cuda.jit((arytype, int32[::1], arytype))(math_pow)
    cfunc[1, nelem](A, B, C)
    Cref = np.empty_like(A)
    for i in range(len(A)):
        Cref[i] = math.pow(A[i], B[i])
    np.testing.assert_allclose(np.power(A, B).astype(npdtype), C, rtol=1e-06)