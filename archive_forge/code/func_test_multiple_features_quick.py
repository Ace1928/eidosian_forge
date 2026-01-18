import re
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
@pytest.mark.parametrize('to_pandas', [True, False])
@pytest.mark.parametrize('smooth', [1.0, 'auto'])
@pytest.mark.parametrize('target_type', ['binary-ints', 'binary-str', 'continuous'])
def test_multiple_features_quick(to_pandas, smooth, target_type):
    """Check target encoder with multiple features."""
    X_ordinal = np.array([[1, 1], [0, 1], [1, 1], [2, 1], [1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.int64)
    if target_type == 'binary-str':
        y_train = np.array(['a', 'b', 'a', 'a', 'b', 'b', 'a', 'b'])
        y_integer = LabelEncoder().fit_transform(y_train)
        cv = StratifiedKFold(2, random_state=0, shuffle=True)
    elif target_type == 'binary-ints':
        y_train = np.array([3, 4, 3, 3, 3, 4, 4, 4])
        y_integer = LabelEncoder().fit_transform(y_train)
        cv = StratifiedKFold(2, random_state=0, shuffle=True)
    else:
        y_train = np.array([3.0, 5.1, 2.4, 3.5, 4.1, 5.5, 10.3, 7.3], dtype=np.float32)
        y_integer = y_train
        cv = KFold(2, random_state=0, shuffle=True)
    y_mean = np.mean(y_integer)
    categories = [[0, 1, 2], [0, 1]]
    X_test = np.array([[0, 1], [3, 0], [1, 10]], dtype=np.int64)
    if to_pandas:
        pd = pytest.importorskip('pandas')
        X_train = pd.DataFrame({'feat0': X_ordinal[:, 0], 'feat1': np.array(['cat', 'dog'], dtype=object)[X_ordinal[:, 1]]})
        X_test = pd.DataFrame({'feat0': X_test[:, 0], 'feat1': ['dog', 'cat', 'snake']})
    else:
        X_train = X_ordinal
    expected_X_fit_transform = np.empty_like(X_ordinal, dtype=np.float64)
    for f_idx, cats in enumerate(categories):
        for train_idx, test_idx in cv.split(X_ordinal, y_integer):
            X_, y_ = (X_ordinal[train_idx, f_idx], y_integer[train_idx])
            current_encoding = _encode_target(X_, y_, len(cats), smooth)
            expected_X_fit_transform[test_idx, f_idx] = current_encoding[X_ordinal[test_idx, f_idx]]
    expected_encodings = []
    for f_idx, cats in enumerate(categories):
        current_encoding = _encode_target(X_ordinal[:, f_idx], y_integer, len(cats), smooth)
        expected_encodings.append(current_encoding)
    expected_X_test_transform = np.array([[expected_encodings[0][0], expected_encodings[1][1]], [y_mean, expected_encodings[1][0]], [expected_encodings[0][1], y_mean]], dtype=np.float64)
    enc = TargetEncoder(smooth=smooth, cv=2, random_state=0)
    X_fit_transform = enc.fit_transform(X_train, y_train)
    assert_allclose(X_fit_transform, expected_X_fit_transform)
    assert len(enc.encodings_) == 2
    for i in range(2):
        assert_allclose(enc.encodings_[i], expected_encodings[i])
    X_test_transform = enc.transform(X_test)
    assert_allclose(X_test_transform, expected_X_test_transform)