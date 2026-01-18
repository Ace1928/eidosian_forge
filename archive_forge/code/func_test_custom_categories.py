import re
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
@pytest.mark.parametrize('X, categories', [(np.array([[0] * 10 + [1] * 10 + [3]], dtype=np.int64).T, [[0, 1, 2]]), (np.array([['cat'] * 10 + ['dog'] * 10 + ['snake']], dtype=object).T, [['dog', 'cat', 'cow']])])
@pytest.mark.parametrize('smooth', [4.0, 'auto'])
def test_custom_categories(X, categories, smooth):
    """Custom categories with unknown categories that are not in training data."""
    rng = np.random.RandomState(0)
    y = rng.uniform(low=-10, high=20, size=X.shape[0])
    enc = TargetEncoder(categories=categories, smooth=smooth, random_state=0).fit(X, y)
    y_mean = y.mean()
    X_trans = enc.transform(X[-1:])
    assert X_trans[0, 0] == pytest.approx(y_mean)
    assert len(enc.encodings_) == 1
    assert enc.encodings_[0][-1] == pytest.approx(y_mean)