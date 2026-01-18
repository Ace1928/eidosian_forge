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
def test_classification_report_multiclass_with_digits():
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)
    expected_report = '              precision    recall  f1-score   support\n\n      setosa    0.82609   0.79167   0.80851        24\n  versicolor    0.33333   0.09677   0.15000        31\n   virginica    0.41860   0.90000   0.57143        20\n\n    accuracy                        0.53333        75\n   macro avg    0.52601   0.59615   0.50998        75\nweighted avg    0.51375   0.53333   0.47310        75\n'
    report = classification_report(y_true, y_pred, labels=np.arange(len(iris.target_names)), target_names=iris.target_names, digits=5)
    assert report == expected_report