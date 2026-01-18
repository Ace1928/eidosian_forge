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
@pytest.mark.parametrize('zero_division, expected_score', [(0, 0), (1, 0.5)])
def test_jaccard_score_zero_division_set_value(zero_division, expected_score):
    y_true = np.array([[1, 0, 1], [0, 0, 0]])
    y_pred = np.array([[0, 0, 0], [0, 0, 0]])
    with warnings.catch_warnings():
        warnings.simplefilter('error', UndefinedMetricWarning)
        score = jaccard_score(y_true, y_pred, average='samples', zero_division=zero_division)
    assert score == pytest.approx(expected_score)