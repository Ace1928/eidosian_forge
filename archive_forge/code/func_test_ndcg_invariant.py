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
def test_ndcg_invariant():
    y_true = np.arange(70).reshape(7, 10)
    y_score = y_true + np.random.RandomState(0).uniform(-0.2, 0.2, size=y_true.shape)
    ndcg = ndcg_score(y_true, y_score)
    ndcg_no_ties = ndcg_score(y_true, y_score, ignore_ties=True)
    assert ndcg == pytest.approx(ndcg_no_ties)
    assert ndcg == pytest.approx(1.0)
    y_score += 1000
    assert ndcg_score(y_true, y_score) == pytest.approx(1.0)