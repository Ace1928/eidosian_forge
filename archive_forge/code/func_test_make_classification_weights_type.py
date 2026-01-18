import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
@pytest.mark.parametrize('weights, err_type, err_msg', [([], ValueError, 'Weights specified but incompatible with number of classes.'), ([0.25, 0.75, 0.1], ValueError, 'Weights specified but incompatible with number of classes.'), (np.array([]), ValueError, 'Weights specified but incompatible with number of classes.'), (np.array([0.25, 0.75, 0.1]), ValueError, 'Weights specified but incompatible with number of classes.'), (np.random.random(3), ValueError, 'Weights specified but incompatible with number of classes.')])
def test_make_classification_weights_type(weights, err_type, err_msg):
    with pytest.raises(err_type, match=err_msg):
        make_classification(weights=weights)