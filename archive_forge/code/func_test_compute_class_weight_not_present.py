import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._testing import assert_almost_equal, assert_array_almost_equal
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.utils.fixes import CSC_CONTAINERS
@pytest.mark.parametrize('y_type, class_weight, classes, err_msg', [('numeric', 'balanced', np.arange(4), 'classes should have valid labels that are in y'), ('numeric', {'label_not_present': 1.0}, np.arange(4), 'The classes, \\[0, 1, 2, 3\\], are not in class_weight'), ('numeric', 'balanced', np.arange(2), 'classes should include all valid labels'), ('numeric', {0: 1.0, 1: 2.0}, np.arange(2), 'classes should include all valid labels'), ('string', {'dogs': 3, 'cat': 2}, np.array(['dog', 'cat']), "The classes, \\['dog'\\], are not in class_weight")])
def test_compute_class_weight_not_present(y_type, class_weight, classes, err_msg):
    y = np.asarray([0, 0, 0, 1, 1, 2]) if y_type == 'numeric' else np.asarray(['dog', 'cat', 'dog'])
    print(y)
    with pytest.raises(ValueError, match=err_msg):
        compute_class_weight(class_weight, classes=classes, y=y)