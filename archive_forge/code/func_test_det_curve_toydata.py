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
@pytest.mark.parametrize('y_true,y_score,expected_fpr,expected_fnr', [([0, 0, 1], [0, 0.5, 1], [0], [0]), ([0, 0, 1], [0, 0.25, 0.5], [0], [0]), ([0, 0, 1], [0.5, 0.75, 1], [0], [0]), ([0, 0, 1], [0.25, 0.5, 0.75], [0], [0]), ([0, 1, 0], [0, 0.5, 1], [0.5], [0]), ([0, 1, 0], [0, 0.25, 0.5], [0.5], [0]), ([0, 1, 0], [0.5, 0.75, 1], [0.5], [0]), ([0, 1, 0], [0.25, 0.5, 0.75], [0.5], [0]), ([0, 1, 1], [0, 0.5, 1], [0.0], [0]), ([0, 1, 1], [0, 0.25, 0.5], [0], [0]), ([0, 1, 1], [0.5, 0.75, 1], [0], [0]), ([0, 1, 1], [0.25, 0.5, 0.75], [0], [0]), ([1, 0, 0], [0, 0.5, 1], [1, 1, 0.5], [0, 1, 1]), ([1, 0, 0], [0, 0.25, 0.5], [1, 1, 0.5], [0, 1, 1]), ([1, 0, 0], [0.5, 0.75, 1], [1, 1, 0.5], [0, 1, 1]), ([1, 0, 0], [0.25, 0.5, 0.75], [1, 1, 0.5], [0, 1, 1]), ([1, 0, 1], [0, 0.5, 1], [1, 1, 0], [0, 0.5, 0.5]), ([1, 0, 1], [0, 0.25, 0.5], [1, 1, 0], [0, 0.5, 0.5]), ([1, 0, 1], [0.5, 0.75, 1], [1, 1, 0], [0, 0.5, 0.5]), ([1, 0, 1], [0.25, 0.5, 0.75], [1, 1, 0], [0, 0.5, 0.5])])
def test_det_curve_toydata(y_true, y_score, expected_fpr, expected_fnr):
    fpr, fnr, _ = det_curve(y_true, y_score)
    assert_allclose(fpr, expected_fpr)
    assert_allclose(fnr, expected_fnr)