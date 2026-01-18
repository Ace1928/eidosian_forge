import numpy as np
from numba import jit
from numba.core import types
from numba.tests.support import TestCase, tag
import unittest
def check_overlap(self, pyfunc, min_ndim, have_k_argument=False):
    N = 4

    def vary_layouts(orig):
        yield orig.copy(order='C')
        yield orig.copy(order='F')
        a = orig[::-1].copy()[::-1]
        assert not a.flags.c_contiguous and (not a.flags.f_contiguous)
        yield a

    def check(pyfunc, cfunc, pydest, cdest, kwargs):
        pyfunc(pydest, pydest, **kwargs)
        cfunc(cdest, cdest, **kwargs)
        self.assertPreciseEqual(pydest, cdest)
    cfunc = jit(nopython=True)(pyfunc)
    for ndim in range(min_ndim, 4):
        shape = (N,) * ndim
        orig = np.arange(0, N ** ndim).reshape(shape)
        for pydest, cdest in zip(vary_layouts(orig), vary_layouts(orig)):
            if have_k_argument:
                for k in range(1, N):
                    check(pyfunc, cfunc, pydest, cdest, dict(k=k))
            else:
                check(pyfunc, cfunc, pydest, cdest, {})