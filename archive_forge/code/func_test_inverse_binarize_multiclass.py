import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_inverse_binarize_multiclass(csr_container):
    got = _inverse_binarize_multiclass(csr_container([[0, 1, 0], [-1, 0, -1], [0, 0, 0]]), np.arange(3))
    assert_array_equal(got, np.array([1, 1, 0]))