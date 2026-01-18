import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction._hashing_fast import transform as _hashing_transform
def test_hasher_alternate_sign():
    X = [list('Thequickbrownfoxjumped')]
    Xt = FeatureHasher(alternate_sign=True, input_type='string').fit_transform(X)
    assert Xt.data.min() < 0 and Xt.data.max() > 0
    Xt = FeatureHasher(alternate_sign=False, input_type='string').fit_transform(X)
    assert Xt.data.min() > 0