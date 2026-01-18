import re
import warnings
from functools import partial
from itertools import chain, permutations, product
import numpy as np
import pytest
from scipy import linalg
from scipy.spatial.distance import hamming as sp_hamming
from scipy.stats import bernoulli
from sklearn import datasets, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
from sklearn.metrics._classification import _check_targets
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.extmath import _nanaverage
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import check_random_state
def test_multilabel_confusion_matrix_errors():
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    with pytest.raises(ValueError, match='inconsistent numbers of samples'):
        multilabel_confusion_matrix(y_true, y_pred, sample_weight=[1, 2])
    with pytest.raises(ValueError, match='should be a 1d array'):
        multilabel_confusion_matrix(y_true, y_pred, sample_weight=[[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    err_msg = 'All labels must be in \\[0, n labels\\)'
    with pytest.raises(ValueError, match=err_msg):
        multilabel_confusion_matrix(y_true, y_pred, labels=[-1])
    err_msg = 'All labels must be in \\[0, n labels\\)'
    with pytest.raises(ValueError, match=err_msg):
        multilabel_confusion_matrix(y_true, y_pred, labels=[3])
    with pytest.raises(ValueError, match='Samplewise metrics'):
        multilabel_confusion_matrix([0, 1, 2], [1, 2, 0], samplewise=True)
    err_msg = 'multiclass-multioutput is not supported'
    with pytest.raises(ValueError, match=err_msg):
        multilabel_confusion_matrix([[0, 1, 2], [2, 1, 0]], [[1, 2, 0], [1, 0, 2]])