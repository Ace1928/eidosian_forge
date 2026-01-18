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
def check_lrap_without_tie_and_increasing_score(lrap_score):
    for n_labels in range(2, 10):
        y_score = n_labels - (np.arange(n_labels).reshape((1, n_labels)) + 1)
        y_true = np.zeros((1, n_labels))
        y_true[0, 0] = 1
        y_true[0, -1] = 1
        assert_almost_equal(lrap_score(y_true, y_score), (2 / n_labels + 1) / 2)
        for n_relevant in range(1, n_labels):
            for pos in range(n_labels - n_relevant):
                y_true = np.zeros((1, n_labels))
                y_true[0, pos:pos + n_relevant] = 1
                assert_almost_equal(lrap_score(y_true, y_score), sum(((r + 1) / ((pos + r + 1) * n_relevant) for r in range(n_relevant))))