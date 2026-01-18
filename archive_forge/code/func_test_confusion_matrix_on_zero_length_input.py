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
@pytest.mark.parametrize('labels', (None, [0, 1], [0, 1, 2]), ids=['None', 'binary', 'multiclass'])
def test_confusion_matrix_on_zero_length_input(labels):
    expected_n_classes = len(labels) if labels else 0
    expected = np.zeros((expected_n_classes, expected_n_classes), dtype=int)
    cm = confusion_matrix([], [], labels=labels)
    assert_array_equal(cm, expected)