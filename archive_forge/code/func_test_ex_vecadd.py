import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout
def test_ex_vecadd(self):
    import numpy as np
    from numba import cuda

    @cuda.jit
    def f(a, b, c):
        tid = cuda.grid(1)
        size = len(c)
        if tid < size:
            c[tid] = a[tid] + b[tid]
    np.random.seed(1)
    N = 100000
    a = cuda.to_device(np.random.random(N))
    b = cuda.to_device(np.random.random(N))
    c = cuda.device_array_like(a)
    f.forall(len(a))(a, b, c)
    print(c.copy_to_host())
    nthreads = 256
    nblocks = len(a) // nthreads + 1
    f[nblocks, nthreads](a, b, c)
    print(c.copy_to_host())
    np.testing.assert_equal(c.copy_to_host(), a.copy_to_host() + b.copy_to_host())