import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
@pytest.mark.parametrize('values, classes, unknown', [(np.array([2, 1, 3, 1, 3], dtype='int64'), np.array([1, 2, 3], dtype='int64'), np.array([4], dtype='int64')), (np.array(['b', 'a', 'c', 'a', 'c'], dtype=object), np.array(['a', 'b', 'c'], dtype=object), np.array(['d'], dtype=object)), (np.array(['b', 'a', 'c', 'a', 'c']), np.array(['a', 'b', 'c']), np.array(['d']))], ids=['int64', 'object', 'str'])
def test_label_encoder(values, classes, unknown):
    le = LabelEncoder()
    le.fit(values)
    assert_array_equal(le.classes_, classes)
    assert_array_equal(le.transform(values), [1, 0, 2, 0, 2])
    assert_array_equal(le.inverse_transform([1, 0, 2, 0, 2]), values)
    le = LabelEncoder()
    ret = le.fit_transform(values)
    assert_array_equal(ret, [1, 0, 2, 0, 2])
    with pytest.raises(ValueError, match='unseen labels'):
        le.transform(unknown)