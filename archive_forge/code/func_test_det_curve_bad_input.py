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
@pytest.mark.parametrize('y_true, y_pred, err_msg', [([0, 1], [0, 0.5, 1], 'inconsistent numbers of samples'), ([0, 1, 1], [0, 0.5], 'inconsistent numbers of samples'), ([0, 0, 0], [0, 0.5, 1], 'Only one class present in y_true'), ([1, 1, 1], [0, 0.5, 1], 'Only one class present in y_true'), (['cancer', 'cancer', 'not cancer'], [0.2, 0.3, 0.8], 'pos_label is not specified')])
def test_det_curve_bad_input(y_true, y_pred, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        det_curve(y_true, y_pred)