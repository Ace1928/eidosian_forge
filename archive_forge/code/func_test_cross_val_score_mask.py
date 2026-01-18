import os
import re
import sys
import tempfile
import warnings
from functools import partial
from io import StringIO
from time import sleep
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import (
from sklearn.model_selection import (
from sklearn.model_selection._validation import (
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.model_selection.tests.test_search import FailingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.svm import SVC, LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.utils import shuffle
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
def test_cross_val_score_mask():
    svm = SVC(kernel='linear')
    iris = load_iris()
    X, y = (iris.data, iris.target)
    kfold = KFold(5)
    scores_indices = cross_val_score(svm, X, y, cv=kfold)
    kfold = KFold(5)
    cv_masks = []
    for train, test in kfold.split(X, y):
        mask_train = np.zeros(len(y), dtype=bool)
        mask_test = np.zeros(len(y), dtype=bool)
        mask_train[train] = 1
        mask_test[test] = 1
        cv_masks.append((train, test))
    scores_masks = cross_val_score(svm, X, y, cv=cv_masks)
    assert_array_equal(scores_indices, scores_masks)