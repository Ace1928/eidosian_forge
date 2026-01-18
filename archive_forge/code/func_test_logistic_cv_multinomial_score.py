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
@pytest.mark.parametrize('scoring, multiclass_agg_list', [('accuracy', ['']), ('precision', ['_macro', '_weighted']), ('f1', ['_macro', '_weighted']), ('neg_log_loss', ['']), ('recall', ['_macro', '_weighted'])])
def test_logistic_cv_multinomial_score(scoring, multiclass_agg_list):
    X, y = make_classification(n_samples=100, random_state=0, n_classes=3, n_informative=6)
    train, test = (np.arange(80), np.arange(80, 100))
    lr = LogisticRegression(C=1.0, multi_class='multinomial')
    params = lr.get_params()
    for key in ['C', 'n_jobs', 'warm_start']:
        del params[key]
    lr.fit(X[train], y[train])
    for averaging in multiclass_agg_list:
        scorer = get_scorer(scoring + averaging)
        assert_array_almost_equal(_log_reg_scoring_path(X, y, train, test, Cs=[1.0], scoring=scorer, pos_class=None, max_squared_sum=None, sample_weight=None, score_params=None, **params)[2][0], scorer(lr, X[test], y[test]))