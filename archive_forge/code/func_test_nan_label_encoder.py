import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
def test_nan_label_encoder():
    """Check that label encoder encodes nans in transform.

    Non-regression test for #22628.
    """
    le = LabelEncoder()
    le.fit(['a', 'a', 'b', np.nan])
    y_trans = le.transform([np.nan])
    assert_array_equal(y_trans, [2])