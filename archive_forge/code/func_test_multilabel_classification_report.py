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
@ignore_warnings
def test_multilabel_classification_report():
    n_classes = 4
    n_samples = 50
    _, y_true = make_multilabel_classification(n_features=1, n_samples=n_samples, n_classes=n_classes, random_state=0)
    _, y_pred = make_multilabel_classification(n_features=1, n_samples=n_samples, n_classes=n_classes, random_state=1)
    expected_report = '              precision    recall  f1-score   support\n\n           0       0.50      0.67      0.57        24\n           1       0.51      0.74      0.61        27\n           2       0.29      0.08      0.12        26\n           3       0.52      0.56      0.54        27\n\n   micro avg       0.50      0.51      0.50       104\n   macro avg       0.45      0.51      0.46       104\nweighted avg       0.45      0.51      0.46       104\n samples avg       0.46      0.42      0.40       104\n'
    report = classification_report(y_true, y_pred)
    assert report == expected_report