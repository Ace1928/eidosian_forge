import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
def test_label_encoder_errors():
    le = LabelEncoder()
    with pytest.raises(ValueError):
        le.transform([])
    with pytest.raises(ValueError):
        le.inverse_transform([])
    le = LabelEncoder()
    le.fit([1, 2, 3, -1, 1])
    msg = 'contains previously unseen labels'
    with pytest.raises(ValueError, match=msg):
        le.inverse_transform([-2])
    with pytest.raises(ValueError, match=msg):
        le.inverse_transform([-2, -3, -4])
    msg = 'should be a 1d array.+shape \\(\\)'
    with pytest.raises(ValueError, match=msg):
        le.inverse_transform('')