import re
import warnings
import numpy as np
import pytest
from scipy import stats
from sklearn import datasets, svm
from sklearn.datasets import make_multilabel_classification
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
from sklearn.metrics._ranking import _dcg_sample_scores, _ndcg_sample_scores
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
from sklearn.utils.extmath import softmax
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import (
def make_prediction(dataset=None, binary=False):
    """Make some classification predictions on a toy dataset using a SVC

    If binary is True restrict to a binary classification problem instead of a
    multiclass classification problem
    """
    if dataset is None:
        dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target
    if binary:
        X, y = (X[y < 2], y[y < 2])
    n_samples, n_features = X.shape
    p = np.arange(n_samples)
    rng = check_random_state(37)
    rng.shuffle(p)
    X, y = (X[p], y[p])
    half = int(n_samples / 2)
    rng = np.random.RandomState(0)
    X = np.c_[X, rng.randn(n_samples, 200 * n_features)]
    clf = svm.SVC(kernel='linear', probability=True, random_state=0)
    y_score = clf.fit(X[:half], y[:half]).predict_proba(X[half:])
    if binary:
        y_score = y_score[:, 1]
    y_pred = clf.predict(X[half:])
    y_true = y[half:]
    return (y_true, y_pred, y_score)