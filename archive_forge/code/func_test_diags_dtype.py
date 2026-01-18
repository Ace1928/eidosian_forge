import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
def test_diags_dtype(self):
    x = construct.diags([2.2], offsets=[0], shape=(2, 2), dtype=int)
    assert_equal(x.dtype, int)
    assert_equal(x.toarray(), [[2, 0], [0, 2]])