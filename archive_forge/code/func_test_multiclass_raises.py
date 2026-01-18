import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.tree import (
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS
@pytest.mark.parametrize('TreeClassifier', TREE_BASED_CLASSIFIER_CLASSES)
def test_multiclass_raises(TreeClassifier):
    X, y = make_classification(n_samples=100, n_features=5, n_classes=3, n_informative=3, random_state=0)
    y[0] = 0
    monotonic_cst = np.zeros(X.shape[1])
    monotonic_cst[0] = -1
    monotonic_cst[1] = 1
    est = TreeClassifier(max_depth=None, monotonic_cst=monotonic_cst, random_state=0)
    msg = 'Monotonicity constraints are not supported with multiclass classification'
    with pytest.raises(ValueError, match=msg):
        est.fit(X, y)