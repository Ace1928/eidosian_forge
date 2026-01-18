import warnings
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets, linear_model
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._least_angle import _lars_path_residues
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
@pytest.mark.parametrize('classifier', [linear_model.Lars, linear_model.LarsCV, linear_model.LassoLarsIC])
def test_lars_precompute(classifier):
    G = np.dot(X.T, X)
    clf = classifier(precompute=G)
    output_1 = ignore_warnings(clf.fit)(X, y).coef_
    for precompute in [True, False, 'auto', None]:
        clf = classifier(precompute=precompute)
        output_2 = clf.fit(X, y).coef_
        assert_array_almost_equal(output_1, output_2, decimal=8)