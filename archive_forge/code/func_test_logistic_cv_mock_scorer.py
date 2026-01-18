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
def test_logistic_cv_mock_scorer():

    class MockScorer:

        def __init__(self):
            self.calls = 0
            self.scores = [0.1, 0.4, 0.8, 0.5]

        def __call__(self, model, X, y, sample_weight=None):
            score = self.scores[self.calls % len(self.scores)]
            self.calls += 1
            return score
    mock_scorer = MockScorer()
    Cs = [1, 2, 3, 4]
    cv = 2
    lr = LogisticRegressionCV(Cs=Cs, scoring=mock_scorer, cv=cv)
    X, y = make_classification(random_state=0)
    lr.fit(X, y)
    assert lr.C_[0] == Cs[2]
    assert mock_scorer.calls == cv * len(Cs)
    mock_scorer.calls = 0
    custom_score = lr.score(X, lr.predict(X))
    assert custom_score == mock_scorer.scores[0]
    assert mock_scorer.calls == 1