import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._testing import assert_almost_equal, assert_array_almost_equal
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.utils.fixes import CSC_CONTAINERS
@pytest.mark.parametrize('y_type, class_weight, indices, err_msg', [('single-output', {1: 2, 2: 1}, range(4), "The only valid class_weight for subsampling is 'balanced'."), ('multi-output', {1: 2, 2: 1}, None, 'For multi-output, class_weight should be a list of dicts, or the string'), ('multi-output', [{1: 2, 2: 1}], None, 'Got 1 element\\(s\\) while having 2 outputs')])
def test_compute_sample_weight_errors(y_type, class_weight, indices, err_msg):
    y_single_output = np.asarray([1, 1, 1, 2, 2, 2])
    y_multi_output = np.asarray([[1, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 1]])
    y = y_single_output if y_type == 'single-output' else y_multi_output
    with pytest.raises(ValueError, match=err_msg):
        compute_sample_weight(class_weight, y, indices=indices)