import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def gen_atomic_extreme_funcs(func):
    fns = dedent('\n    def atomic(res, ary):\n        tx = cuda.threadIdx.x\n        bx = cuda.blockIdx.x\n        {func}(res, 0, ary[tx, bx])\n\n    def atomic_double_normalizedindex(res, ary):\n        tx = cuda.threadIdx.x\n        bx = cuda.blockIdx.x\n        {func}(res, 0, ary[tx, uint64(bx)])\n\n    def atomic_double_oneindex(res, ary):\n        tx = cuda.threadIdx.x\n        {func}(res, 0, ary[tx])\n\n    def atomic_double_shared(res, ary):\n        tid = cuda.threadIdx.x\n        smary = cuda.shared.array(32, float64)\n        smary[tid] = ary[tid]\n        smres = cuda.shared.array(1, float64)\n        if tid == 0:\n            smres[0] = res[0]\n        cuda.syncthreads()\n        {func}(smres, 0, smary[tid])\n        cuda.syncthreads()\n        if tid == 0:\n            res[0] = smres[0]\n    ').format(func=func)
    ld = {}
    exec(fns, {'cuda': cuda, 'float64': float64, 'uint64': uint64}, ld)
    return (ld['atomic'], ld['atomic_double_normalizedindex'], ld['atomic_double_oneindex'], ld['atomic_double_shared'])