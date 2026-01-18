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
@ignore_warnings
@pytest.mark.parametrize('coo_container', COO_CONTAINERS)
def test_multilabel_representation_invariance(coo_container):
    n_classes = 4
    n_samples = 50
    _, y1 = make_multilabel_classification(n_features=1, n_classes=n_classes, random_state=0, n_samples=n_samples, allow_unlabeled=True)
    _, y2 = make_multilabel_classification(n_features=1, n_classes=n_classes, random_state=1, n_samples=n_samples, allow_unlabeled=True)
    y1 = np.vstack([y1, [[0] * n_classes]])
    y2 = np.vstack([y2, [[0] * n_classes]])
    y1_sparse_indicator = coo_container(y1)
    y2_sparse_indicator = coo_container(y2)
    y1_list_array_indicator = list(y1)
    y2_list_array_indicator = list(y2)
    y1_list_list_indicator = [list(a) for a in y1_list_array_indicator]
    y2_list_list_indicator = [list(a) for a in y2_list_array_indicator]
    for name in MULTILABELS_METRICS:
        metric = ALL_METRICS[name]
        if isinstance(metric, partial):
            metric.__module__ = 'tmp'
            metric.__name__ = name
        measure = metric(y1, y2)
        assert_allclose(metric(y1_sparse_indicator, y2_sparse_indicator), measure, err_msg='%s failed representation invariance between dense and sparse indicator formats.' % name)
        assert_almost_equal(metric(y1_list_list_indicator, y2_list_list_indicator), measure, err_msg='%s failed representation invariance  between dense array and list of list indicator formats.' % name)
        assert_almost_equal(metric(y1_list_array_indicator, y2_list_array_indicator), measure, err_msg='%s failed representation invariance  between dense and list of array indicator formats.' % name)