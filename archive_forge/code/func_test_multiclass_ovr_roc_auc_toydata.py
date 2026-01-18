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
@pytest.mark.parametrize('y_true, labels', [(np.array([0, 1, 2, 2]), None), (['a', 'b', 'c', 'c'], None), ([0, 1, 2, 2], [0, 1, 2]), (['a', 'b', 'c', 'c'], ['a', 'b', 'c'])])
def test_multiclass_ovr_roc_auc_toydata(y_true, labels):
    y_scores = np.array([[1.0, 0.0, 0.0], [0.1, 0.5, 0.4], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]])
    out_0 = roc_auc_score([1, 0, 0, 0], y_scores[:, 0])
    out_1 = roc_auc_score([0, 1, 0, 0], y_scores[:, 1])
    out_2 = roc_auc_score([0, 0, 1, 1], y_scores[:, 2])
    assert_almost_equal(roc_auc_score(y_true, y_scores, multi_class='ovr', labels=labels, average=None), [out_0, out_1, out_2])
    result_unweighted = (out_0 + out_1 + out_2) / 3.0
    assert_almost_equal(roc_auc_score(y_true, y_scores, multi_class='ovr', labels=labels), result_unweighted)
    result_weighted = out_0 * 0.25 + out_1 * 0.25 + out_2 * 0.5
    assert_almost_equal(roc_auc_score(y_true, y_scores, multi_class='ovr', labels=labels, average='weighted'), result_weighted)