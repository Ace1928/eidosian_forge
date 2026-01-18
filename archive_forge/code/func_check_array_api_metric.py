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
def check_array_api_metric(metric, array_namespace, device, dtype_name, y_true_np, y_pred_np, sample_weight):
    xp = _array_api_for_tests(array_namespace, device)
    y_true_xp = xp.asarray(y_true_np, device=device)
    y_pred_xp = xp.asarray(y_pred_np, device=device)
    metric_np = metric(y_true_np, y_pred_np, sample_weight=sample_weight)
    if sample_weight is not None:
        sample_weight = xp.asarray(sample_weight, device=device)
    with config_context(array_api_dispatch=True):
        metric_xp = metric(y_true_xp, y_pred_xp, sample_weight=sample_weight)
        assert_allclose(metric_xp, metric_np, atol=_atol_for_type(dtype_name))