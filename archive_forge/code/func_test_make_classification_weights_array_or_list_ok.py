import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
@pytest.mark.parametrize('kwargs', [{}, {'n_classes': 3, 'n_informative': 3}])
def test_make_classification_weights_array_or_list_ok(kwargs):
    X1, y1 = make_classification(weights=[0.1, 0.9], random_state=0, **kwargs)
    X2, y2 = make_classification(weights=np.array([0.1, 0.9]), random_state=0, **kwargs)
    assert_almost_equal(X1, X2)
    assert_almost_equal(y1, y2)