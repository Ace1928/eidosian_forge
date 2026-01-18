import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
def test_diags_bad(self):
    a = array([1, 2, 3, 4, 5])
    b = array([6, 7, 8, 9, 10])
    c = array([11, 12, 13, 14, 15])
    cases = []
    cases.append(([a[:0]], 0, (1, 1)))
    cases.append(([a[:4], b, c[:3]], [-1, 0, 1], (5, 5)))
    cases.append(([a[:2], c, b[:3]], [-4, 2, -1], (6, 5)))
    cases.append(([a[:2], c, b[:3]], [-4, 2, -1], None))
    cases.append(([], [-4, 2, -1], None))
    cases.append(([1], [-5], (4, 4)))
    cases.append(([a], 0, None))
    for d, o, shape in cases:
        assert_raises(ValueError, construct.diags, d, offsets=o, shape=shape)
    assert_raises(TypeError, construct.diags, [[None]], offsets=[0])