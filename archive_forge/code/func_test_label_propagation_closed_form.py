import warnings
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.semi_supervised import _label_propagation as label_propagation
from sklearn.utils._testing import (
def test_label_propagation_closed_form(global_dtype):
    n_classes = 2
    X, y = make_classification(n_classes=n_classes, n_samples=200, random_state=0)
    X = X.astype(global_dtype, copy=False)
    y[::3] = -1
    Y = np.zeros((len(y), n_classes + 1))
    Y[np.arange(len(y)), y] = 1
    unlabelled_idx = Y[:, (-1,)].nonzero()[0]
    labelled_idx = (Y[:, (-1,)] == 0).nonzero()[0]
    clf = label_propagation.LabelPropagation(max_iter=100, tol=1e-10, gamma=0.1)
    clf.fit(X, y)
    T_bar = clf._build_graph()
    Tuu = T_bar[tuple(np.meshgrid(unlabelled_idx, unlabelled_idx, indexing='ij'))]
    Tul = T_bar[tuple(np.meshgrid(unlabelled_idx, labelled_idx, indexing='ij'))]
    Y = Y[:, :-1]
    Y_l = Y[labelled_idx, :]
    Y_u = np.dot(np.dot(np.linalg.inv(np.eye(Tuu.shape[0]) - Tuu), Tul), Y_l)
    expected = Y.copy()
    expected[unlabelled_idx, :] = Y_u
    expected /= expected.sum(axis=1)[:, np.newaxis]
    assert_allclose(expected, clf.label_distributions_, atol=0.0001)