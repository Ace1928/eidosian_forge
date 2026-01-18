import itertools
import os
import warnings
from functools import partial
import numpy as np
import pytest
from numpy.testing import (
from scipy import sparse
from sklearn import config_context
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.metrics import get_scorer, log_loss
from sklearn.model_selection import (
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.svm import l1_min_c
from sklearn.utils import _IS_32BIT, compute_class_weight, shuffle
from sklearn.utils._testing import ignore_warnings, skip_if_no_parallel
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('C', [0.001, 0.1, 1, 10, 100, 1000, 1000000.0])
@pytest.mark.parametrize('penalty, l1_ratio', [('l1', 1), ('l2', 0)])
def test_elastic_net_l1_l2_equivalence(C, penalty, l1_ratio):
    X, y = make_classification(random_state=0)
    lr_enet = LogisticRegression(penalty='elasticnet', C=C, l1_ratio=l1_ratio, solver='saga', random_state=0, tol=0.01)
    lr_expected = LogisticRegression(penalty=penalty, C=C, solver='saga', random_state=0, tol=0.01)
    lr_enet.fit(X, y)
    lr_expected.fit(X, y)
    assert_array_almost_equal(lr_enet.coef_, lr_expected.coef_)