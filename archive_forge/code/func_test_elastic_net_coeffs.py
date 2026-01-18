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
def test_elastic_net_coeffs():
    X, y = make_classification(random_state=0)
    C = 2.0
    l1_ratio = 0.5
    coeffs = list()
    for penalty, ratio in (('elasticnet', l1_ratio), ('l1', None), ('l2', None)):
        lr = LogisticRegression(penalty=penalty, C=C, solver='saga', random_state=0, l1_ratio=ratio, tol=0.001, max_iter=200)
        lr.fit(X, y)
        coeffs.append(lr.coef_)
    elastic_net_coeffs, l1_coeffs, l2_coeffs = coeffs
    assert not np.allclose(elastic_net_coeffs, l1_coeffs, rtol=0, atol=0.1)
    assert not np.allclose(elastic_net_coeffs, l2_coeffs, rtol=0, atol=0.1)
    assert not np.allclose(l2_coeffs, l1_coeffs, rtol=0, atol=0.1)