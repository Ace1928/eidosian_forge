import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction._hashing_fast import transform as _hashing_transform
def test_hash_collisions():
    X = [list('Thequickbrownfoxjumped')]
    Xt = FeatureHasher(alternate_sign=True, n_features=1, input_type='string').fit_transform(X)
    assert abs(Xt.data[0]) < len(X[0])
    Xt = FeatureHasher(alternate_sign=False, n_features=1, input_type='string').fit_transform(X)
    assert Xt.data[0] == len(X[0])