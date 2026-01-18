from functools import partial
from inspect import signature
from itertools import chain, permutations, product
import numpy as np
import pytest
from sklearn._config import config_context
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import (
from sklearn.metrics._base import _average_binary_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples, check_random_state
@pytest.mark.parametrize('metric', CLASSIFICATION_METRICS.values())
def test_classification_binary_continuous_input(metric):
    """check that classification metrics raise a message of mixed type data
    with continuous/binary target vectors."""
    y_true, y_score = (['a', 'b', 'a'], [0.1, 0.2, 0.3])
    err_msg = "Classification metrics can't handle a mix of binary and continuous targets"
    with pytest.raises(ValueError, match=err_msg):
        metric(y_true, y_score)