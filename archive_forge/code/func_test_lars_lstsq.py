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
@pytest.mark.filterwarnings('ignore: `rcond` parameter will change')
def test_lars_lstsq():
    X1 = 3 * X
    clf = linear_model.LassoLars(alpha=0.0)
    clf.fit(X1, y)
    coef_lstsq = np.linalg.lstsq(X1, y, rcond=None)[0]
    assert_array_almost_equal(clf.coef_, coef_lstsq)