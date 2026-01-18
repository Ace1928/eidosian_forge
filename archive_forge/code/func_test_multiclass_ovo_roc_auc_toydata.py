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
@pytest.mark.parametrize('y_true, labels', [(np.array([0, 1, 0, 2]), [0, 1, 2]), (np.array([0, 1, 0, 2]), None), (['a', 'b', 'a', 'c'], ['a', 'b', 'c']), (['a', 'b', 'a', 'c'], None)])
def test_multiclass_ovo_roc_auc_toydata(y_true, labels):
    y_scores = np.array([[0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.35, 0.5, 0.15], [0, 0.2, 0.8]])
    score_01 = roc_auc_score([1, 0, 1], [0.1, 0.3, 0.35])
    score_10 = roc_auc_score([0, 1, 0], [0.8, 0.4, 0.5])
    average_score_01 = (score_01 + score_10) / 2
    score_02 = roc_auc_score([1, 1, 0], [0.1, 0.35, 0])
    score_20 = roc_auc_score([0, 0, 1], [0.1, 0.15, 0.8])
    average_score_02 = (score_02 + score_20) / 2
    score_12 = roc_auc_score([1, 0], [0.4, 0.2])
    score_21 = roc_auc_score([0, 1], [0.3, 0.8])
    average_score_12 = (score_12 + score_21) / 2
    ovo_unweighted_score = (average_score_01 + average_score_02 + average_score_12) / 3
    assert_almost_equal(roc_auc_score(y_true, y_scores, labels=labels, multi_class='ovo'), ovo_unweighted_score)
    pair_scores = [average_score_01, average_score_02, average_score_12]
    prevalence = [0.75, 0.75, 0.5]
    ovo_weighted_score = np.average(pair_scores, weights=prevalence)
    assert_almost_equal(roc_auc_score(y_true, y_scores, labels=labels, multi_class='ovo', average='weighted'), ovo_weighted_score)
    error_message = "average=None is not implemented for multi_class='ovo'."
    with pytest.raises(NotImplementedError, match=error_message):
        roc_auc_score(y_true, y_scores, labels=labels, multi_class='ovo', average=None)