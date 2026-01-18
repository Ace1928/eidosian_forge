import re
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
@pytest.mark.parametrize('categories, unknown_values', [([np.array([0, 1, 2], dtype=np.int64)], 'auto'), ([np.array(['cat', 'dog', 'snake'], dtype=object)], ['bear', 'rabbit'])])
@pytest.mark.parametrize('target_labels', [np.array([1, 2, 3]), np.array(['a', 'b', 'c'])])
@pytest.mark.parametrize('smooth', [5.0, 'auto'])
def test_encoding_multiclass(global_random_seed, categories, unknown_values, target_labels, smooth):
    """Check encoding for multiclass targets."""
    rng = np.random.RandomState(global_random_seed)
    n_samples = 80
    n_features = 2
    feat_1_int = np.array(rng.randint(low=0, high=2, size=n_samples))
    feat_2_int = np.array(rng.randint(low=0, high=3, size=n_samples))
    feat_1 = categories[0][feat_1_int]
    feat_2 = categories[0][feat_2_int]
    X_train = np.column_stack((feat_1, feat_2))
    X_train_int = np.column_stack((feat_1_int, feat_2_int))
    categories_ = [[0, 1], [0, 1, 2]]
    n_classes = 3
    y_train_int = np.array(rng.randint(low=0, high=n_classes, size=n_samples))
    y_train = target_labels[y_train_int]
    y_train_enc = LabelBinarizer().fit_transform(y_train)
    n_splits = 3
    cv = StratifiedKFold(n_splits=n_splits, random_state=global_random_seed, shuffle=True)
    expected_X_fit_transform = np.empty((X_train_int.shape[0], X_train_int.shape[1] * n_classes), dtype=np.float64)
    for f_idx, cats in enumerate(categories_):
        for c_idx in range(n_classes):
            for train_idx, test_idx in cv.split(X_train, y_train):
                y_class = y_train_enc[:, c_idx]
                X_, y_ = (X_train_int[train_idx, f_idx], y_class[train_idx])
                current_encoding = _encode_target(X_, y_, len(cats), smooth)
                exp_idx = c_idx + f_idx * n_classes
                expected_X_fit_transform[test_idx, exp_idx] = current_encoding[X_train_int[test_idx, f_idx]]
    target_encoder = TargetEncoder(smooth=smooth, cv=n_splits, random_state=global_random_seed)
    X_fit_transform = target_encoder.fit_transform(X_train, y_train)
    assert target_encoder.target_type_ == 'multiclass'
    assert_allclose(X_fit_transform, expected_X_fit_transform)
    expected_encodings = []
    for f_idx, cats in enumerate(categories_):
        for c_idx in range(n_classes):
            y_class = y_train_enc[:, c_idx]
            current_encoding = _encode_target(X_train_int[:, f_idx], y_class, len(cats), smooth)
            expected_encodings.append(current_encoding)
    assert len(target_encoder.encodings_) == n_features * n_classes
    for i in range(n_features * n_classes):
        assert_allclose(target_encoder.encodings_[i], expected_encodings[i])
    assert_array_equal(target_encoder.classes_, target_labels)
    X_test_int = np.array([[0, 1], [1, 2], [4, 5]])
    if unknown_values == 'auto':
        X_test = X_test_int
    else:
        X_test = np.empty_like(X_test_int[:-1, :], dtype=object)
        for column_idx in range(X_test_int.shape[1]):
            X_test[:, column_idx] = categories[0][X_test_int[:-1, column_idx]]
        X_test = np.vstack((X_test, unknown_values))
    y_mean = np.mean(y_train_enc, axis=0)
    expected_X_test_transform = np.empty((X_test_int.shape[0], X_test_int.shape[1] * n_classes), dtype=np.float64)
    n_rows = X_test_int.shape[0]
    f_idx = [0, 0, 0, 1, 1, 1]
    for row_idx in range(n_rows - 1):
        for i, enc in enumerate(expected_encodings):
            expected_X_test_transform[row_idx, i] = enc[X_test_int[row_idx, f_idx[i]]]
    mean_idx = [0, 1, 2, 0, 1, 2]
    for i in range(n_classes * n_features):
        expected_X_test_transform[n_rows - 1, i] = y_mean[mean_idx[i]]
    X_test_transform = target_encoder.transform(X_test)
    assert_allclose(X_test_transform, expected_X_test_transform)