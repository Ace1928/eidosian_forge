import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def test_cdist_out(self, metric):
    eps = 1e-15
    X1 = eo['cdist-X1']
    X2 = eo['cdist-X2']
    out_r, out_c = (X1.shape[0], X2.shape[0])
    kwargs = dict()
    if metric == 'minkowski':
        kwargs['p'] = 1.23
    out1 = np.empty((out_r, out_c), dtype=np.float64)
    Y1 = cdist(X1, X2, metric, **kwargs)
    Y2 = cdist(X1, X2, metric, out=out1, **kwargs)
    assert_allclose(Y1, Y2, rtol=eps, verbose=verbose > 2)
    assert_(Y2 is out1)
    out2 = np.empty((out_r - 1, out_c + 1), dtype=np.float64)
    with pytest.raises(ValueError):
        cdist(X1, X2, metric, out=out2, **kwargs)
    out3 = np.empty((2 * out_r, 2 * out_c), dtype=np.float64)[::2, ::2]
    out4 = np.empty((out_r, out_c), dtype=np.float64, order='F')
    with pytest.raises(ValueError):
        cdist(X1, X2, metric, out=out3, **kwargs)
    with pytest.raises(ValueError):
        cdist(X1, X2, metric, out=out4, **kwargs)
    out5 = np.empty((out_r, out_c), dtype=np.int64)
    with pytest.raises(ValueError):
        cdist(X1, X2, metric, out=out5, **kwargs)