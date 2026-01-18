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
@pytest.mark.parametrize('categorical_features, array_type', [(['col_A', 'col_C'], 'dataframe'), ([0, 2], 'array'), ([True, False, True], 'array')])
def test_grid_resolution_with_categorical(pyplot, categorical_features, array_type):
    """Check that we raise a ValueError when the grid_resolution is too small
    respect to the number of categories in the categorical features targeted.
    """
    X = [['A', 1, 'A'], ['B', 0, 'C'], ['C', 2, 'B']]
    column_name = ['col_A', 'col_B', 'col_C']
    X = _convert_container(X, array_type, columns_name=column_name)
    y = np.array([1.2, 0.5, 0.45]).T
    preprocessor = make_column_transformer((OneHotEncoder(), categorical_features))
    model = make_pipeline(preprocessor, LinearRegression())
    model.fit(X, y)
    err_msg = 'resolution of the computed grid is less than the minimum number of categories'
    with pytest.raises(ValueError, match=err_msg):
        PartialDependenceDisplay.from_estimator(model, X, features=['col_C'], feature_names=column_name, categorical_features=categorical_features, grid_resolution=2)