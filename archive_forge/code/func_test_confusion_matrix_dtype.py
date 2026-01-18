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
def test_confusion_matrix_dtype():
    y = [0, 1, 1]
    weight = np.ones(len(y))
    cm = confusion_matrix(y, y)
    assert cm.dtype == np.int64
    for dtype in [np.bool_, np.int32, np.uint64]:
        cm = confusion_matrix(y, y, sample_weight=weight.astype(dtype, copy=False))
        assert cm.dtype == np.int64
    for dtype in [np.float32, np.float64, None, object]:
        cm = confusion_matrix(y, y, sample_weight=weight.astype(dtype, copy=False))
        assert cm.dtype == np.float64
    weight = np.full(len(y), 4294967295, dtype=np.uint32)
    cm = confusion_matrix(y, y, sample_weight=weight)
    assert cm[0, 0] == 4294967295
    assert cm[1, 1] == 8589934590
    weight = np.full(len(y), 9223372036854775807, dtype=np.int64)
    cm = confusion_matrix(y, y, sample_weight=weight)
    assert cm[0, 0] == 9223372036854775807
    assert cm[1, 1] == -2