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
def check_array_api_multiclass_classification_metric(metric, array_namespace, device, dtype_name):
    y_true_np = np.array([0, 1, 2, 3])
    y_pred_np = np.array([0, 1, 0, 2])
    check_array_api_metric(metric, array_namespace, device, dtype_name, y_true_np=y_true_np, y_pred_np=y_pred_np, sample_weight=None)
    sample_weight = np.array([0.0, 0.1, 2.0, 1.0], dtype=dtype_name)
    check_array_api_metric(metric, array_namespace, device, dtype_name, y_true_np=y_true_np, y_pred_np=y_pred_np, sample_weight=sample_weight)