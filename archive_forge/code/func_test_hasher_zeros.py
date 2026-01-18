import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction._hashing_fast import transform as _hashing_transform
def test_hasher_zeros():
    X = FeatureHasher().transform([{'foo': 0}])
    assert X.data.shape == (0,)