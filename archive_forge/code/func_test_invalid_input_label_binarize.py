import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import datasets
from sklearn.preprocessing._label import (
from sklearn.utils import _to_object_array
from sklearn.utils._testing import assert_array_equal, ignore_warnings
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import type_of_target
def test_invalid_input_label_binarize():
    with pytest.raises(ValueError):
        label_binarize([0, 2], classes=[0, 2], pos_label=0, neg_label=1)
    with pytest.raises(ValueError, match='continuous target data is not '):
        label_binarize([1.2, 2.7], classes=[0, 1])
    with pytest.raises(ValueError, match='mismatch with the labels'):
        label_binarize([[1, 3]], classes=[1, 2, 3])