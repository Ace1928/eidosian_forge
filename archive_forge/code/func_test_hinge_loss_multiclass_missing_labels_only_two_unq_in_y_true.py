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
def test_hinge_loss_multiclass_missing_labels_only_two_unq_in_y_true():
    pred_decision = np.array([[+0.36, -0.17, -0.58], [-0.15, -0.58, -0.48], [-1.45, -0.58, -0.38], [-0.55, -0.78, -0.42], [-1.45, -0.58, -0.38]])
    y_true = np.array([0, 2, 2, 0, 2])
    labels = np.array([0, 1, 2])
    dummy_losses = np.array([1 - pred_decision[0][0] + pred_decision[0][1], 1 - pred_decision[1][2] + pred_decision[1][0], 1 - pred_decision[2][2] + pred_decision[2][1], 1 - pred_decision[3][0] + pred_decision[3][2], 1 - pred_decision[4][2] + pred_decision[4][1]])
    np.clip(dummy_losses, 0, None, out=dummy_losses)
    dummy_hinge_loss = np.mean(dummy_losses)
    assert_almost_equal(hinge_loss(y_true, pred_decision, labels=labels), dummy_hinge_loss)