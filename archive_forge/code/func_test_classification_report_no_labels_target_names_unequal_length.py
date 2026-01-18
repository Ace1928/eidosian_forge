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
def test_classification_report_no_labels_target_names_unequal_length():
    y_true = [0, 0, 2, 0, 0]
    y_pred = [0, 2, 2, 0, 0]
    target_names = ['class 0', 'class 1', 'class 2']
    err_msg = 'Number of classes, 2, does not match size of target_names, 3. Try specifying the labels parameter'
    with pytest.raises(ValueError, match=err_msg):
        classification_report(y_true, y_pred, target_names=target_names)