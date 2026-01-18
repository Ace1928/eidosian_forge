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
def test_lrap_sample_weighting_zero_labels():
    y_true = np.array([[1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]], dtype=bool)
    y_score = np.array([[0.3, 0.4, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    samplewise_lraps = np.array([0.5, 0.75, 1.0])
    sample_weight = np.array([1.0, 1.0, 0.0])
    assert_almost_equal(label_ranking_average_precision_score(y_true, y_score, sample_weight=sample_weight), np.sum(sample_weight * samplewise_lraps) / np.sum(sample_weight))