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
def test_jaccard_score_validation():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1])
    err_msg = 'pos_label=2 is not a valid label. It should be one of \\[0, 1\\]'
    with pytest.raises(ValueError, match=err_msg):
        jaccard_score(y_true, y_pred, average='binary', pos_label=2)
    y_true = np.array([[0, 1, 1], [1, 0, 0]])
    y_pred = np.array([[1, 1, 1], [1, 0, 1]])
    msg1 = "Target is multilabel-indicator but average='binary'. Please choose another average setting, one of \\[None, 'micro', 'macro', 'weighted', 'samples'\\]."
    with pytest.raises(ValueError, match=msg1):
        jaccard_score(y_true, y_pred, average='binary', pos_label=-1)
    y_true = np.array([0, 1, 1, 0, 2])
    y_pred = np.array([1, 1, 1, 1, 0])
    msg2 = "Target is multiclass but average='binary'. Please choose another average setting, one of \\[None, 'micro', 'macro', 'weighted'\\]."
    with pytest.raises(ValueError, match=msg2):
        jaccard_score(y_true, y_pred, average='binary')
    msg3 = 'Samplewise metrics are not available outside of multilabel classification.'
    with pytest.raises(ValueError, match=msg3):
        jaccard_score(y_true, y_pred, average='samples')
    msg = "Note that pos_label \\(set to 3\\) is ignored when average != 'binary' \\(got 'micro'\\). You may use labels=\\[pos_label\\] to specify a single positive class."
    with pytest.warns(UserWarning, match=msg):
        jaccard_score(y_true, y_pred, average='micro', pos_label=3)