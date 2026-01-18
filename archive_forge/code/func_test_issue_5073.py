from numba import cuda, int32, float64, void
from numba.core.errors import TypingError
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
import numpy as np
from numba.np import numpy_support as nps
from .extensions_usecases import test_struct_model_type, TestStruct
def test_issue_5073(self):
    arr = np.arange(1024)
    nelem = len(arr)
    nthreads = 16
    nblocks = int(nelem / nthreads)
    dt = nps.from_dtype(arr.dtype)
    nshared = nthreads * arr.dtype.itemsize
    chunksize = int(nthreads / 2)

    @cuda.jit
    def sm_slice_copy(x, y, chunksize):
        dynsmem = cuda.shared.array(0, dtype=dt)
        sm1 = dynsmem[0:chunksize]
        sm2 = dynsmem[chunksize:chunksize * 2]
        tx = cuda.threadIdx.x
        bx = cuda.blockIdx.x
        bd = cuda.blockDim.x
        i = bx * bd + tx
        if i < len(x):
            if tx < chunksize:
                sm1[tx] = x[i]
            else:
                sm2[tx - chunksize] = x[i]
        cuda.syncthreads()
        if tx == 0:
            for j in range(chunksize):
                y[bd * bx + j] = sm1[j]
                y[bd * bx + j + chunksize] = sm2[j]
    d_result = cuda.device_array_like(arr)
    sm_slice_copy[nblocks, nthreads, 0, nshared](arr, d_result, chunksize)
    host_result = d_result.copy_to_host()
    np.testing.assert_array_equal(arr, host_result)