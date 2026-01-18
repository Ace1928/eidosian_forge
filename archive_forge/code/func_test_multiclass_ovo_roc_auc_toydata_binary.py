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
@pytest.mark.parametrize('y_true, labels', [(np.array([0, 2, 0, 2]), [0, 1, 2]), (np.array(['a', 'd', 'a', 'd']), ['a', 'b', 'd'])])
def test_multiclass_ovo_roc_auc_toydata_binary(y_true, labels):
    y_scores = np.array([[0.2, 0.0, 0.8], [0.6, 0.0, 0.4], [0.55, 0.0, 0.45], [0.4, 0.0, 0.6]])
    score_01 = roc_auc_score([1, 0, 1, 0], [0.2, 0.6, 0.55, 0.4])
    score_10 = roc_auc_score([0, 1, 0, 1], [0.8, 0.4, 0.45, 0.6])
    ovo_score = (score_01 + score_10) / 2
    assert_almost_equal(roc_auc_score(y_true, y_scores, labels=labels, multi_class='ovo'), ovo_score)
    assert_almost_equal(roc_auc_score(y_true, y_scores, labels=labels, multi_class='ovo', average='weighted'), ovo_score)