import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_einsum_misc(self):
    a = np.ones((1, 2))
    b = np.ones((2, 2, 1))
    assert_equal(np.einsum('ij...,j...->i...', a, b), [[[2], [2]]])
    assert_equal(np.einsum('ij...,j...->i...', a, b, optimize=True), [[[2], [2]]])
    assert_equal(np.einsum('ij...,j...->i...', a, b), [[[2], [2]]])
    assert_equal(np.einsum('...i,...i', [1, 2, 3], [2, 3, 4]), 20)
    assert_equal(np.einsum('...i,...i', [1, 2, 3], [2, 3, 4], optimize='greedy'), 20)
    a = np.ones((5, 12, 4, 2, 3), np.int64)
    b = np.ones((5, 12, 11), np.int64)
    assert_equal(np.einsum('ijklm,ijn,ijn->', a, b, b), np.einsum('ijklm,ijn->', a, b))
    assert_equal(np.einsum('ijklm,ijn,ijn->', a, b, b, optimize=True), np.einsum('ijklm,ijn->', a, b, optimize=True))
    a = np.arange(1, 3)
    b = np.arange(1, 5).reshape(2, 2)
    c = np.arange(1, 9).reshape(4, 2)
    assert_equal(np.einsum('x,yx,zx->xzy', a, b, c), [[[1, 3], [3, 9], [5, 15], [7, 21]], [[8, 16], [16, 32], [24, 48], [32, 64]]])
    assert_equal(np.einsum('x,yx,zx->xzy', a, b, c, optimize=True), [[[1, 3], [3, 9], [5, 15], [7, 21]], [[8, 16], [16, 32], [24, 48], [32, 64]]])
    assert_equal(np.einsum('i,j', [1], [2], out=None), [[2]])