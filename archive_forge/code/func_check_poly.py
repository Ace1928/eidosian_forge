import numpy as np
from numpy.testing import assert_, assert_allclose
import pytest
from scipy.special import _ufuncs
import scipy.special._orthogonal as orth
from scipy.special._testutils import FuncData
def check_poly(self, func, param_ranges=[], x_range=[], nn=10, nparam=10, nx=10, rtol=1e-08):
    np.random.seed(1234)
    dataset = []
    for n in np.arange(nn):
        params = [a + (b - a) * np.random.rand(nparam) for a, b in param_ranges]
        params = np.asarray(params).T
        if not param_ranges:
            params = [0]
        for p in params:
            if param_ranges:
                p = (n,) + tuple(p)
            else:
                p = (n,)
            x = x_range[0] + (x_range[1] - x_range[0]) * np.random.rand(nx)
            x[0] = x_range[0]
            x[1] = x_range[1]
            kw = dict(sig=(len(p) + 1) * 'd' + '->d')
            z = np.c_[np.tile(p, (nx, 1)), x, func(*p + (x,), **kw)]
            dataset.append(z)
    dataset = np.concatenate(dataset, axis=0)

    def polyfunc(*p):
        p = (p[0].astype(int),) + p[1:]
        kw = dict(sig='l' + (len(p) - 1) * 'd' + '->d')
        return func(*p, **kw)
    with np.errstate(all='raise'):
        ds = FuncData(polyfunc, dataset, list(range(len(param_ranges) + 2)), -1, rtol=rtol)
        ds.check()