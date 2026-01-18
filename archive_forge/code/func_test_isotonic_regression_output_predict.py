import copy
import pickle
import warnings
import numpy as np
import pytest
from scipy.special import expit
import sklearn
from sklearn.datasets import make_regression
from sklearn.isotonic import (
from sklearn.utils import shuffle
from sklearn.utils._testing import (
from sklearn.utils.validation import check_array
def test_isotonic_regression_output_predict():
    """Check that `predict` does return the expected output type.

    We need to check that `transform` will output a DataFrame and a NumPy array
    when we set `transform_output` to `pandas`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/25499
    """
    pd = pytest.importorskip('pandas')
    X, y = make_regression(n_samples=10, n_features=1, random_state=42)
    regressor = IsotonicRegression()
    with sklearn.config_context(transform_output='pandas'):
        regressor.fit(X, y)
        X_trans = regressor.transform(X)
        y_pred = regressor.predict(X)
    assert isinstance(X_trans, pd.DataFrame)
    assert isinstance(y_pred, np.ndarray)