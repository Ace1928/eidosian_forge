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
def test_label_ranking_avg_precision_score_should_allow_csr_matrix_for_y_true_input(csr_container):
    y_true = csr_container([[1, 0, 0], [0, 0, 1]])
    y_score = np.array([[0.5, 0.9, 0.6], [0, 0, 1]])
    result = label_ranking_average_precision_score(y_true, y_score)
    assert result == pytest.approx(2 / 3)