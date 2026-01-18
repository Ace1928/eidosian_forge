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
def test_lars_cv_max_iter(recwarn):
    warnings.simplefilter('always')
    with np.errstate(divide='raise', invalid='raise'):
        X = diabetes.data
        y = diabetes.target
        rng = np.random.RandomState(42)
        x = rng.randn(len(y))
        X = diabetes.data
        X = np.c_[X, x, x]
        X = StandardScaler().fit_transform(X)
        lars_cv = linear_model.LassoLarsCV(max_iter=5, cv=5)
        lars_cv.fit(X, y)
    recorded_warnings = [str(w) for w in recwarn]
    assert len(recorded_warnings) == 0