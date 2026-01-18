import array
import cmath
from functools import reduce
import itertools
from operator import mul
import math
import symengine as se
from symengine.test_utilities import raises
from symengine import have_numpy
import unittest
from unittest.case import SkipTest
@unittest.skipUnless(have_numpy, 'Numpy not installed')
def test_Lambdify_gh174():
    args = x, y = se.symbols('x y')
    nargs = len(args)
    vec1 = se.DenseMatrix([x, x ** 2, x ** 3])
    assert vec1.shape == (3, 1)
    assert np.asarray(vec1).shape == (3, 1)
    lmb1 = se.Lambdify([x], vec1)
    out1 = lmb1(3)
    assert out1.shape == (3, 1)
    assert np.all(out1 == [[3], [9], [27]])
    assert lmb1([2, 3]).shape == (2, 3, 1)
    lmb1.order = 'F'
    out1a = lmb1([2, 3])
    assert out1a.shape == (3, 1, 2)
    ref1a_squeeze = [[2, 3], [4, 9], [8, 27]]
    assert np.all(out1a.squeeze() == ref1a_squeeze)
    assert out1a.flags['F_CONTIGUOUS']
    assert not out1a.flags['C_CONTIGUOUS']
    lmb2c = se.Lambdify(args, vec1, x + y, order='C')
    lmb2f = se.Lambdify(args, vec1, x + y, order='F')
    for out2a in [lmb2c([2, 3]), lmb2f([2, 3])]:
        assert np.all(out2a[0] == [[2], [4], [8]])
        assert out2a[0].ndim == 2
        assert out2a[1] == 5
        assert out2a[1].ndim == 0
    inp2b = np.array([[2.0, 3.0], [1.0, 2.0], [0.0, 6.0]])
    raises(ValueError, lambda: lmb2c(inp2b.T))
    out2c = lmb2c(inp2b)
    out2f = lmb2f(np.asfortranarray(inp2b.T))
    assert out2c[0].shape == (3, 3, 1)
    assert out2f[0].shape == (3, 1, 3)
    for idx, (_x, _y) in enumerate(inp2b):
        assert np.all(out2c[0][idx, ...] == [[_x], [_x ** 2], [_x ** 3]])
    assert np.all(out2c[1] == [5, 3, 6])
    assert np.all(out2f[1] == [5, 3, 6])
    assert out2c[1].shape == (3,)
    assert out2f[1].shape == (3,)

    def _mtx3(_x, _y):
        return [[_x ** row_idx + _y ** col_idx for col_idx in range(3)] for row_idx in range(4)]
    mtx3c = np.array(_mtx3(x, y), order='C')
    mtx3f = np.array(_mtx3(x, y), order='F')
    lmb3c = se.Lambdify([x, y], x * y, mtx3c, vec1, order='C')
    lmb3f = se.Lambdify([x, y], x * y, mtx3f, vec1, order='F')
    inp3c = np.array([[2.0, 3], [3, 4], [5, 7], [6, 2], [3, 1]])
    inp3f = np.asfortranarray(inp3c.T)
    raises(ValueError, lambda: lmb3c(inp3c.T))
    out3c = lmb3c(inp3c)
    assert out3c[0].shape == (5,)
    assert out3c[1].shape == (5, 4, 3)
    assert out3c[2].shape == (5, 3, 1)
    for a, b in zip(out3c, lmb3c(np.ravel(inp3c))):
        assert np.all(a == b)
    out3f = lmb3f(inp3f)
    assert out3f[0].shape == (5,)
    assert out3f[1].shape == (4, 3, 5)
    assert out3f[2].shape == (3, 1, 5)
    for a, b in zip(out3f, lmb3f(np.ravel(inp3f, order='F'))):
        assert np.all(a == b)
    for idx, (_x, _y) in enumerate(inp3c):
        assert out3c[0][idx] == _x * _y
        assert out3f[0][idx] == _x * _y
        assert np.all(out3c[1][idx, ...] == _mtx3(_x, _y))
        assert np.all(out3f[1][..., idx] == _mtx3(_x, _y))
        assert np.all(out3c[2][idx, ...] == [[_x], [_x ** 2], [_x ** 3]])
        assert np.all(out3f[2][..., idx] == [[_x], [_x ** 2], [_x ** 3]])