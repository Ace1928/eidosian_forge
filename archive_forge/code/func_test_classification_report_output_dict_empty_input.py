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
def test_classification_report_output_dict_empty_input():
    report = classification_report(y_true=[], y_pred=[], output_dict=True)
    expected_report = {'accuracy': 0.0, 'macro avg': {'f1-score': np.nan, 'precision': np.nan, 'recall': np.nan, 'support': 0}, 'weighted avg': {'f1-score': np.nan, 'precision': np.nan, 'recall': np.nan, 'support': 0}}
    assert isinstance(report, dict)
    assert report.keys() == expected_report.keys()
    for key in expected_report:
        if key == 'accuracy':
            assert isinstance(report[key], float)
            assert report[key] == expected_report[key]
        else:
            assert report[key].keys() == expected_report[key].keys()
            for metric in expected_report[key]:
                assert_almost_equal(expected_report[key][metric], report[key][metric])