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
def test_average_precision_score_non_binary_class():
    """Test multiclass-multiouptut for `average_precision_score`."""
    y_true = np.array([[2, 2, 1], [1, 2, 0], [0, 1, 2], [1, 2, 1], [2, 0, 1], [1, 2, 1]])
    y_score = np.array([[0.7, 0.2, 0.1], [0.4, 0.3, 0.3], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.1, 0.2, 0.7]])
    err_msg = 'multiclass-multioutput format is not supported'
    with pytest.raises(ValueError, match=err_msg):
        average_precision_score(y_true, y_score, pos_label=2)