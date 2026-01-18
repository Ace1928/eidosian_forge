import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
def test_multilabel_binarizer_inverse_validation():
    inp = [(1, 1, 1, 0)]
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(inp)
    with pytest.raises(ValueError):
        mlb.inverse_transform(np.array([[1, 3]]))
    mlb.inverse_transform(np.array([[0, 0]]))
    mlb.inverse_transform(np.array([[1, 1]]))
    mlb.inverse_transform(np.array([[1, 0]]))
    with pytest.raises(ValueError):
        mlb.inverse_transform(np.array([[1]]))
    with pytest.raises(ValueError):
        mlb.inverse_transform(np.array([[1, 1, 1]]))