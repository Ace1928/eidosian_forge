import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
def test_diags(self):
    a = array([1, 2, 3, 4, 5])
    b = array([6, 7, 8, 9, 10])
    c = array([11, 12, 13, 14, 15])
    cases = []
    cases.append((a[:1], 0, (1, 1), [[1]]))
    cases.append(([a[:1]], [0], (1, 1), [[1]]))
    cases.append(([a[:1]], [0], (2, 1), [[1], [0]]))
    cases.append(([a[:1]], [0], (1, 2), [[1, 0]]))
    cases.append(([a[:1]], [1], (1, 2), [[0, 1]]))
    cases.append(([a[:2]], [0], (2, 2), [[1, 0], [0, 2]]))
    cases.append(([a[:1]], [-1], (2, 2), [[0, 0], [1, 0]]))
    cases.append(([a[:3]], [0], (3, 4), [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0]]))
    cases.append(([a[:3]], [1], (3, 4), [[0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3]]))
    cases.append(([a[:1]], [-2], (3, 5), [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]))
    cases.append(([a[:2]], [-1], (3, 5), [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 2, 0, 0, 0]]))
    cases.append(([a[:3]], [0], (3, 5), [[1, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 3, 0, 0]]))
    cases.append(([a[:3]], [1], (3, 5), [[0, 1, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 3, 0]]))
    cases.append(([a[:3]], [2], (3, 5), [[0, 0, 1, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 3]]))
    cases.append(([a[:2]], [3], (3, 5), [[0, 0, 0, 1, 0], [0, 0, 0, 0, 2], [0, 0, 0, 0, 0]]))
    cases.append(([a[:1]], [4], (3, 5), [[0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))
    cases.append(([a[:1]], [-4], (5, 3), [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]]))
    cases.append(([a[:2]], [-3], (5, 3), [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 2, 0]]))
    cases.append(([a[:3]], [-2], (5, 3), [[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]]))
    cases.append(([a[:3]], [-1], (5, 3), [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [0, 0, 0]]))
    cases.append(([a[:3]], [0], (5, 3), [[1, 0, 0], [0, 2, 0], [0, 0, 3], [0, 0, 0], [0, 0, 0]]))
    cases.append(([a[:2]], [1], (5, 3), [[0, 1, 0], [0, 0, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]]))
    cases.append(([a[:1]], [2], (5, 3), [[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]))
    cases.append(([a[:3], b[:1]], [0, 2], (3, 3), [[1, 0, 6], [0, 2, 0], [0, 0, 3]]))
    cases.append(([a[:2], b[:3]], [-1, 0], (3, 4), [[6, 0, 0, 0], [1, 7, 0, 0], [0, 2, 8, 0]]))
    cases.append(([a[:4], b[:3]], [2, -3], (6, 6), [[0, 0, 1, 0, 0, 0], [0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 3, 0], [6, 0, 0, 0, 0, 4], [0, 7, 0, 0, 0, 0], [0, 0, 8, 0, 0, 0]]))
    cases.append(([a[:4], b, c[:4]], [-1, 0, 1], (5, 5), [[6, 11, 0, 0, 0], [1, 7, 12, 0, 0], [0, 2, 8, 13, 0], [0, 0, 3, 9, 14], [0, 0, 0, 4, 10]]))
    cases.append(([a[:2], b[:3], c], [-4, 2, -1], (6, 5), [[0, 0, 6, 0, 0], [11, 0, 0, 7, 0], [0, 12, 0, 0, 8], [0, 0, 13, 0, 0], [1, 0, 0, 14, 0], [0, 2, 0, 0, 15]]))
    cases.append(([a], [0], (1, 1), [[1]]))
    cases.append(([a[:3], b], [0, 2], (3, 3), [[1, 0, 6], [0, 2, 0], [0, 0, 3]]))
    cases.append((np.array([[1, 2, 3], [4, 5, 6]]), [0, -1], (3, 3), [[1, 0, 0], [4, 2, 0], [0, 5, 3]]))
    cases.append(([1, -2, 1], [1, 0, -1], (3, 3), [[-2, 1, 0], [1, -2, 1], [0, 1, -2]]))
    for d, o, shape, result in cases:
        err_msg = f'{d!r} {o!r} {shape!r} {result!r}'
        assert_equal(construct.diags(d, offsets=o, shape=shape).toarray(), result, err_msg=err_msg)
        if shape[0] == shape[1] and hasattr(d[0], '__len__') and (len(d[0]) <= max(shape)):
            assert_equal(construct.diags(d, offsets=o).toarray(), result, err_msg=err_msg)