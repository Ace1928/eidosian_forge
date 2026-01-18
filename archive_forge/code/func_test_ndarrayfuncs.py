import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
def test_ndarrayfuncs(self):
    d = np.arange(24.0).reshape((2, 3, 4))
    m = np.zeros(24, dtype=bool).reshape((2, 3, 4))
    m[:, :, -1] = True
    a = np.ma.array(d, mask=m)

    def testaxis(f, a, d):
        numpy_f = numpy.__getattribute__(f)
        ma_f = np.ma.__getattribute__(f)
        assert_equal(ma_f(a, axis=1)[..., :-1], numpy_f(d[..., :-1], axis=1))
        assert_equal(ma_f(a, axis=(0, 1))[..., :-1], numpy_f(d[..., :-1], axis=(0, 1)))

    def testkeepdims(f, a, d):
        numpy_f = numpy.__getattribute__(f)
        ma_f = np.ma.__getattribute__(f)
        assert_equal(ma_f(a, keepdims=True).shape, numpy_f(d, keepdims=True).shape)
        assert_equal(ma_f(a, keepdims=False).shape, numpy_f(d, keepdims=False).shape)
        assert_equal(ma_f(a, axis=1, keepdims=True)[..., :-1], numpy_f(d[..., :-1], axis=1, keepdims=True))
        assert_equal(ma_f(a, axis=(0, 1), keepdims=True)[..., :-1], numpy_f(d[..., :-1], axis=(0, 1), keepdims=True))
    for f in ['sum', 'prod', 'mean', 'var', 'std']:
        testaxis(f, a, d)
        testkeepdims(f, a, d)
    for f in ['min', 'max']:
        testaxis(f, a, d)
    d = np.arange(24).reshape((2, 3, 4)) % 2 == 0
    a = np.ma.array(d, mask=m)
    for f in ['all', 'any']:
        testaxis(f, a, d)
        testkeepdims(f, a, d)