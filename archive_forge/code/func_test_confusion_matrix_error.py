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
@pytest.mark.parametrize('labels, err_msg', [([], "'labels' should contains at least one label."), ([3, 4], 'At least one label specified must be in y_true')], ids=['empty list', 'unknown labels'])
def test_confusion_matrix_error(labels, err_msg):
    y_true, y_pred, _ = make_prediction(binary=False)
    with pytest.raises(ValueError, match=err_msg):
        confusion_matrix(y_true, y_pred, labels=labels)