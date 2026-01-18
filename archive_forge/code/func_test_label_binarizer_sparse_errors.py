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
def test_label_binarizer_sparse_errors(csr_container):
    err_msg = 'foo format is not supported'
    with pytest.raises(ValueError, match=err_msg):
        _inverse_binarize_thresholding(y=csr_container([[1, 2], [2, 1]]), output_type='foo', classes=[1, 2], threshold=0)
    err_msg = 'The number of class is not equal to the number of dimension of y.'
    with pytest.raises(ValueError, match=err_msg):
        _inverse_binarize_thresholding(y=csr_container([[1, 2], [2, 1]]), output_type='foo', classes=[1, 2, 3], threshold=0)