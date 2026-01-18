import re
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
@pytest.mark.parametrize('smooth', [0.0, 1000.0, 'auto'])
def test_invariance_of_encoding_under_label_permutation(smooth, global_random_seed):
    rng = np.random.RandomState(global_random_seed)
    y = rng.normal(size=1000)
    n_categories = 30
    X = KBinsDiscretizer(n_bins=n_categories, encode='ordinal').fit_transform(y.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=global_random_seed)
    permutated_labels = rng.permutation(n_categories)
    X_train_permuted = permutated_labels[X_train.astype(np.int32)]
    X_test_permuted = permutated_labels[X_test.astype(np.int32)]
    target_encoder = TargetEncoder(smooth=smooth, random_state=global_random_seed)
    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    X_test_encoded = target_encoder.transform(X_test)
    X_train_permuted_encoded = target_encoder.fit_transform(X_train_permuted, y_train)
    X_test_permuted_encoded = target_encoder.transform(X_test_permuted)
    assert_allclose(X_train_encoded, X_train_permuted_encoded)
    assert_allclose(X_test_encoded, X_test_permuted_encoded)