import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats.mstats import mquantiles
from sklearn.compose import make_column_transformer
from sklearn.datasets import (
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._testing import _convert_container
@pytest.fixture(scope='module')
def clf_diabetes(diabetes):
    clf = GradientBoostingRegressor(n_estimators=10, random_state=1)
    clf.fit(diabetes.data, diabetes.target)
    return clf