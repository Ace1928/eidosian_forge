import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
@pytest.mark.parametrize('eye', [construct.eye, construct.eye_array])
def test_eye_one(self, eye):
    assert_equal(eye(1).toarray(), [[1]])
    assert_equal(eye(2).toarray(), [[1, 0], [0, 1]])
    I = eye(3, dtype='int8', format='dia')
    assert_equal(I.dtype, np.dtype('int8'))
    assert_equal(I.format, 'dia')
    for fmt in sparse_formats:
        I = eye(3, format=fmt)
        assert_equal(I.format, fmt)
        assert_equal(I.toarray(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])