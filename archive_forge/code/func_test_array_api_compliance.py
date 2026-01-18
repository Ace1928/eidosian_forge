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
@pytest.mark.parametrize('array_namespace, device, dtype_name', yield_namespace_device_dtype_combinations())
@pytest.mark.parametrize('metric, check_func', yield_metric_checker_combinations())
def test_array_api_compliance(metric, array_namespace, device, dtype_name, check_func):
    check_func(metric, array_namespace, device, dtype_name)