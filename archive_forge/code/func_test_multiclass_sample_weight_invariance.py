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
@pytest.mark.parametrize('name', sorted(set(ALL_METRICS) - set(REGRESSION_METRICS) - METRICS_WITHOUT_SAMPLE_WEIGHT - METRIC_UNDEFINED_BINARY_MULTICLASS))
def test_multiclass_sample_weight_invariance(name):
    n_samples = 50
    random_state = check_random_state(0)
    y_true = random_state.randint(0, 5, size=(n_samples,))
    y_pred = random_state.randint(0, 5, size=(n_samples,))
    y_score = random_state.random_sample(size=(n_samples, 5))
    metric = ALL_METRICS[name]
    if name in THRESHOLDED_METRICS:
        temp = np.exp(-y_score)
        y_score_norm = temp / temp.sum(axis=-1).reshape(-1, 1)
        check_sample_weight_invariance(name, metric, y_true, y_score_norm)
    else:
        check_sample_weight_invariance(name, metric, y_true, y_pred)