import re
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
def test_fit_transform_not_associated_with_y_if_ordinal_categorical_is_not(global_random_seed):
    cardinality = 30
    n_samples = 3000
    rng = np.random.RandomState(global_random_seed)
    y_train = rng.normal(size=n_samples)
    X_train = rng.randint(0, cardinality, size=n_samples).reshape(-1, 1)
    y_sorted_indices = y_train.argsort()
    y_train = y_train[y_sorted_indices]
    X_train = X_train[y_sorted_indices]
    target_encoder = TargetEncoder(shuffle=True, random_state=global_random_seed)
    X_encoded_train_shuffled = target_encoder.fit_transform(X_train, y_train)
    target_encoder = TargetEncoder(shuffle=False)
    X_encoded_train_no_shuffled = target_encoder.fit_transform(X_train, y_train)
    regressor = RandomForestRegressor(n_estimators=10, min_samples_leaf=20, random_state=global_random_seed)
    cv = ShuffleSplit(n_splits=50, random_state=global_random_seed)
    assert cross_val_score(regressor, X_train, y_train, cv=cv).mean() < 0.1
    assert cross_val_score(regressor, X_encoded_train_shuffled, y_train, cv=cv).mean() < 0.1
    assert cross_val_score(regressor, X_encoded_train_no_shuffled, y_train, cv=cv).mean() > 0.5