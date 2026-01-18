import re
import warnings
import numpy as np
import pytest
from scipy import stats
from sklearn import datasets, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
from sklearn.metrics._ranking import _dcg_sample_scores, _ndcg_sample_scores
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
from sklearn.utils.extmath import softmax
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import (
def check_alternative_lrap_implementation(lrap_score, n_classes=5, n_samples=20, random_state=0):
    _, y_true = make_multilabel_classification(n_features=1, allow_unlabeled=False, random_state=random_state, n_classes=n_classes, n_samples=n_samples)
    y_score = _sparse_random_matrix(n_components=y_true.shape[0], n_features=y_true.shape[1], random_state=random_state)
    if hasattr(y_score, 'toarray'):
        y_score = y_score.toarray()
    score_lrap = label_ranking_average_precision_score(y_true, y_score)
    score_my_lrap = _my_lrap(y_true, y_score)
    assert_almost_equal(score_lrap, score_my_lrap)
    random_state = check_random_state(random_state)
    y_score = random_state.uniform(size=(n_samples, n_classes))
    score_lrap = label_ranking_average_precision_score(y_true, y_score)
    score_my_lrap = _my_lrap(y_true, y_score)
    assert_almost_equal(score_lrap, score_my_lrap)