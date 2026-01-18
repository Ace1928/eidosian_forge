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
def test_multilabel_zero_one_loss_subset():
    y1 = np.array([[0, 1, 1], [1, 0, 1]])
    y2 = np.array([[0, 0, 1], [1, 0, 1]])
    assert zero_one_loss(y1, y2) == 0.5
    assert zero_one_loss(y1, y1) == 0
    assert zero_one_loss(y2, y2) == 0
    assert zero_one_loss(y2, np.logical_not(y2)) == 1
    assert zero_one_loss(y1, np.logical_not(y1)) == 1
    assert zero_one_loss(y1, np.zeros(y1.shape)) == 1
    assert zero_one_loss(y2, np.zeros(y1.shape)) == 1