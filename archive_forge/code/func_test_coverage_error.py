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
def test_coverage_error():
    assert_almost_equal(coverage_error([[0, 1]], [[0.25, 0.75]]), 1)
    assert_almost_equal(coverage_error([[0, 1]], [[0.75, 0.25]]), 2)
    assert_almost_equal(coverage_error([[1, 1]], [[0.75, 0.25]]), 2)
    assert_almost_equal(coverage_error([[0, 0]], [[0.75, 0.25]]), 0)
    assert_almost_equal(coverage_error([[0, 0, 0]], [[0.25, 0.5, 0.75]]), 0)
    assert_almost_equal(coverage_error([[0, 0, 1]], [[0.25, 0.5, 0.75]]), 1)
    assert_almost_equal(coverage_error([[0, 1, 0]], [[0.25, 0.5, 0.75]]), 2)
    assert_almost_equal(coverage_error([[0, 1, 1]], [[0.25, 0.5, 0.75]]), 2)
    assert_almost_equal(coverage_error([[1, 0, 0]], [[0.25, 0.5, 0.75]]), 3)
    assert_almost_equal(coverage_error([[1, 0, 1]], [[0.25, 0.5, 0.75]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 0]], [[0.25, 0.5, 0.75]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 1]], [[0.25, 0.5, 0.75]]), 3)
    assert_almost_equal(coverage_error([[0, 0, 0]], [[0.75, 0.5, 0.25]]), 0)
    assert_almost_equal(coverage_error([[0, 0, 1]], [[0.75, 0.5, 0.25]]), 3)
    assert_almost_equal(coverage_error([[0, 1, 0]], [[0.75, 0.5, 0.25]]), 2)
    assert_almost_equal(coverage_error([[0, 1, 1]], [[0.75, 0.5, 0.25]]), 3)
    assert_almost_equal(coverage_error([[1, 0, 0]], [[0.75, 0.5, 0.25]]), 1)
    assert_almost_equal(coverage_error([[1, 0, 1]], [[0.75, 0.5, 0.25]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 0]], [[0.75, 0.5, 0.25]]), 2)
    assert_almost_equal(coverage_error([[1, 1, 1]], [[0.75, 0.5, 0.25]]), 3)
    assert_almost_equal(coverage_error([[0, 0, 0]], [[0.5, 0.75, 0.25]]), 0)
    assert_almost_equal(coverage_error([[0, 0, 1]], [[0.5, 0.75, 0.25]]), 3)
    assert_almost_equal(coverage_error([[0, 1, 0]], [[0.5, 0.75, 0.25]]), 1)
    assert_almost_equal(coverage_error([[0, 1, 1]], [[0.5, 0.75, 0.25]]), 3)
    assert_almost_equal(coverage_error([[1, 0, 0]], [[0.5, 0.75, 0.25]]), 2)
    assert_almost_equal(coverage_error([[1, 0, 1]], [[0.5, 0.75, 0.25]]), 3)
    assert_almost_equal(coverage_error([[1, 1, 0]], [[0.5, 0.75, 0.25]]), 2)
    assert_almost_equal(coverage_error([[1, 1, 1]], [[0.5, 0.75, 0.25]]), 3)
    assert_almost_equal(coverage_error([[0, 1, 0], [1, 1, 0]], [[0.1, 10.0, -3], [0, 1, 3]]), (1 + 3) / 2.0)
    assert_almost_equal(coverage_error([[0, 1, 0], [1, 1, 0], [0, 1, 1]], [[0.1, 10, -3], [0, 1, 3], [0, 2, 0]]), (1 + 3 + 3) / 3.0)
    assert_almost_equal(coverage_error([[0, 1, 0], [1, 1, 0], [0, 1, 1]], [[0.1, 10, -3], [3, 1, 3], [0, 2, 0]]), (1 + 3 + 3) / 3.0)