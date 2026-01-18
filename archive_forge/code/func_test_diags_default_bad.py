import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
def test_diags_default_bad(self):
    a = array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    assert_raises(ValueError, construct.diags, a)