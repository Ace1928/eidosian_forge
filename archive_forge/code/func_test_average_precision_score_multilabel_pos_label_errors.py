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
def test_average_precision_score_multilabel_pos_label_errors():
    y_true = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    err_msg = 'Parameter pos_label is fixed to 1 for multilabel-indicator y_true. Do not set pos_label or set pos_label to 1.'
    with pytest.raises(ValueError, match=err_msg):
        average_precision_score(y_true, y_pred, pos_label=0)