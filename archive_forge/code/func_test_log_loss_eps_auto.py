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
def test_log_loss_eps_auto(global_dtype):
    """Check the behaviour of `eps="auto"` that changes depending on the input
    array dtype.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/24315
    """
    y_true = np.array([0, 1], dtype=global_dtype)
    y_pred = y_true.copy()
    loss = log_loss(y_true, y_pred, eps='auto')
    assert np.isfinite(loss)