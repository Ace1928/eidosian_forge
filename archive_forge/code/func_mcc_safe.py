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
def mcc_safe(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    true_pos = conf_matrix[1, 1]
    false_pos = conf_matrix[1, 0]
    false_neg = conf_matrix[0, 1]
    n_points = len(y_true)
    pos_rate = (true_pos + false_neg) / n_points
    activity = (true_pos + false_pos) / n_points
    mcc_numerator = true_pos / n_points - pos_rate * activity
    mcc_denominator = activity * pos_rate * (1 - activity) * (1 - pos_rate)
    return mcc_numerator / np.sqrt(mcc_denominator)