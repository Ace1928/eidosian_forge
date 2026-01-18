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
def test_hinge_loss_multiclass_missing_labels_with_labels_none():
    y_true = np.array([0, 1, 2, 2])
    pred_decision = np.array([[+1.27, 0.034, -0.68, -1.4], [-1.45, -0.58, -0.38, -0.17], [-2.36, -0.79, -0.27, +0.24], [-2.36, -0.79, -0.27, +0.24]])
    error_message = 'Please include all labels in y_true or pass labels as third argument'
    with pytest.raises(ValueError, match=error_message):
        hinge_loss(y_true, pred_decision)