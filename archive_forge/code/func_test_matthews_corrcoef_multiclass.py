import re
import warnings
from functools import partial
from itertools import chain, permutations, product
import numpy as np
import pytest
from scipy import linalg
from scipy.spatial.distance import hamming as sp_hamming
from scipy.stats import bernoulli
from sklearn import datasets, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
from sklearn.metrics._classification import _check_targets
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.extmath import _nanaverage
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import check_random_state
def test_matthews_corrcoef_multiclass():
    rng = np.random.RandomState(0)
    ord_a = ord('a')
    n_classes = 4
    y_true = [chr(ord_a + i) for i in rng.randint(0, n_classes, size=20)]
    assert_almost_equal(matthews_corrcoef(y_true, y_true), 1.0)
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred_bad = [2, 2, 0, 0, 1, 1]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred_bad), -0.5)
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred_min = [1, 1, 0, 0, 0, 0]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred_min), -12 / np.sqrt(24 * 16))
    y_true = [0, 1, 2]
    y_pred = [3, 3, 3]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred), 0.0)
    y_true = [3, 3, 3]
    y_pred = [0, 1, 2]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred), 0.0)
    y_1 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_2 = [1, 1, 1, 2, 2, 2, 0, 0, 0]
    assert_almost_equal(matthews_corrcoef(y_1, y_2), 0.0)
    y_true = [0, 0, 1, 1, 2]
    y_pred = [1, 1, 0, 0, 2]
    sample_weight = [1, 1, 1, 1, 0]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight), -1)
    y_true = [0, 0, 1, 2]
    y_pred = [0, 0, 1, 2]
    sample_weight = [1, 1, 0, 0]
    assert_almost_equal(matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight), 0.0)