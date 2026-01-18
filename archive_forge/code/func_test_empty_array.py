from numpy.testing import assert_array_almost_equal, assert_allclose, assert_
from numpy import (array, eye, zeros, empty_like, empty, tril_indices_from,
from numpy.random import rand, randint, seed
from scipy.linalg import ldl
from scipy._lib._util import ComplexWarning
import pytest
from pytest import raises as assert_raises, warns
def test_empty_array():
    a = empty((0, 0), dtype=complex)
    l, d, p = ldl(empty((0, 0)))
    assert_array_almost_equal(l, empty_like(a))
    assert_array_almost_equal(d, empty_like(a))
    assert_array_almost_equal(p, array([], dtype=int))