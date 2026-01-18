import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def check_cas(self, n, fill, unfill, dtype, cas_func, ndim=1):
    res = [fill] * (n // 2) + [unfill] * (n // 2)
    np.random.shuffle(res)
    res = np.asarray(res, dtype=dtype)
    if ndim == 2:
        res.shape = (10, -1)
    out = np.zeros_like(res)
    ary = np.random.randint(1, 10, size=res.shape).astype(res.dtype)
    fill_mask = res == fill
    unfill_mask = res == unfill
    expect_res = np.zeros_like(res)
    expect_res[fill_mask] = ary[fill_mask]
    expect_res[unfill_mask] = unfill
    expect_out = res.copy()
    cuda_func = cuda.jit(cas_func)
    if ndim == 1:
        cuda_func[10, 10](res, out, ary, fill)
    else:
        cuda_func[(10, 10), (10, 10)](res, out, ary, fill)
    np.testing.assert_array_equal(expect_res, res)
    np.testing.assert_array_equal(expect_out, out)