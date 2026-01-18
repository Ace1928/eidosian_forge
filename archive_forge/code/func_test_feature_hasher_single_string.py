import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction._hashing_fast import transform as _hashing_transform
@pytest.mark.parametrize('raw_X', [['my_string', 'another_string'], (x for x in ['my_string', 'another_string'])], ids=['list', 'generator'])
def test_feature_hasher_single_string(raw_X):
    """FeatureHasher raises error when a sample is a single string.

    Non-regression test for gh-13199.
    """
    msg = 'Samples can not be a single string'
    feature_hasher = FeatureHasher(n_features=10, input_type='string')
    with pytest.raises(ValueError, match=msg):
        feature_hasher.transform(raw_X)