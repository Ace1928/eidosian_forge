import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
def test_label_encoder_negative_ints():
    le = LabelEncoder()
    le.fit([1, 1, 4, 5, -1, 0])
    assert_array_equal(le.classes_, [-1, 0, 1, 4, 5])
    assert_array_equal(le.transform([0, 1, 4, 4, 5, -1, -1]), [1, 2, 3, 3, 4, 0, 0])
    assert_array_equal(le.inverse_transform([1, 2, 3, 3, 4, 0, 0]), [0, 1, 4, 4, 5, -1, -1])
    with pytest.raises(ValueError):
        le.transform([0, 6])