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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_label_ranking_loss_sparse(csr_container):
    assert_almost_equal(label_ranking_loss(csr_container(np.array([[0, 1, 0], [1, 1, 0]])), [[0.1, 10, -3], [3, 1, 3]]), (0 + 2 / 2) / 2.0)