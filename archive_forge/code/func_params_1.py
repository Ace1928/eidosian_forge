import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
def params_1(f, bounded=False):
    a = 5.0
    b = np.arange(2.0, 12.0)
    c = np.arange(2.0, 102.0).reshape((10, 10))
    d = np.arange(2.0, 1002.0).reshape((10, 10, 10))
    e = np.array([2.0, 3.0])
    g = np.arange(2.0, 12.0).reshape((1, 10, 1))
    if bounded:
        a = 0.5
        b = b / (1.5 * b.max())
        c = c / (1.5 * c.max())
        d = d / (1.5 * d.max())
        e = e / (1.5 * e.max())
        g = g / (1.5 * g.max())
    f(a)
    f(a, size=(10, 10))
    f(b)
    f(c)
    f(d)
    f(b, size=10)
    f(e, size=(10, 2))
    f(g, size=(10, 10, 10))