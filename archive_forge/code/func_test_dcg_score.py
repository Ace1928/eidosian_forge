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
def test_dcg_score():
    _, y_true = make_multilabel_classification(random_state=0, n_classes=10)
    y_score = -y_true + 1
    _test_dcg_score_for(y_true, y_score)
    y_true, y_score = np.random.RandomState(0).random_sample((2, 100, 10))
    _test_dcg_score_for(y_true, y_score)