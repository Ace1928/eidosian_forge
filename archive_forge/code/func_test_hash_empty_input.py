import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction._hashing_fast import transform as _hashing_transform
def test_hash_empty_input():
    n_features = 16
    raw_X = [[], (), iter(range(0))]
    feature_hasher = FeatureHasher(n_features=n_features, input_type='string')
    X = feature_hasher.transform(raw_X)
    assert_array_equal(X.toarray(), np.zeros((len(raw_X), n_features)))