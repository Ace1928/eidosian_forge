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
@pytest.mark.parametrize('metric_name', CLASSIFICATION_METRICS)
def test_metrics_consistent_type_error(metric_name):
    rng = np.random.RandomState(42)
    y1 = np.array(['spam'] * 3 + ['eggs'] * 2, dtype=object)
    y2 = rng.randint(0, 2, size=y1.size)
    err_msg = 'Labels in y_true and y_pred should be of the same type.'
    with pytest.raises(TypeError, match=err_msg):
        CLASSIFICATION_METRICS[metric_name](y1, y2)