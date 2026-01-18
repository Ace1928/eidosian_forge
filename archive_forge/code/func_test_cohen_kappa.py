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
def test_cohen_kappa():
    y1 = np.array([0] * 40 + [1] * 60)
    y2 = np.array([0] * 20 + [1] * 20 + [0] * 10 + [1] * 50)
    kappa = cohen_kappa_score(y1, y2)
    assert_almost_equal(kappa, 0.348, decimal=3)
    assert kappa == cohen_kappa_score(y2, y1)
    y1 = np.append(y1, [2] * 4)
    y2 = np.append(y2, [2] * 4)
    assert cohen_kappa_score(y1, y2, labels=[0, 1]) == kappa
    assert_almost_equal(cohen_kappa_score(y1, y1), 1.0)
    y1 = np.array([0] * 46 + [1] * 44 + [2] * 10)
    y2 = np.array([0] * 52 + [1] * 32 + [2] * 16)
    assert_almost_equal(cohen_kappa_score(y1, y2), 0.8013, decimal=4)
    y1 = np.array([0] * 46 + [1] * 44 + [2] * 10)
    y2 = np.array([0] * 50 + [1] * 40 + [2] * 10)
    assert_almost_equal(cohen_kappa_score(y1, y2), 0.9315, decimal=4)
    assert_almost_equal(cohen_kappa_score(y1, y2, weights='linear'), 0.9412, decimal=4)
    assert_almost_equal(cohen_kappa_score(y1, y2, weights='quadratic'), 0.9541, decimal=4)