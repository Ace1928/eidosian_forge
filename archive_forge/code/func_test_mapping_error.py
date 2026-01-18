from random import Random
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
def test_mapping_error():
    error_value = "Unsupported value type <class 'dict'> for foo: {'one': 1, 'three': 3}.\nMapping objects are not supported."
    D2 = [{'foo': '1', 'bar': '2'}, {'foo': '3', 'baz': '1'}, {'foo': {'one': 1, 'three': 3}}]
    v = DictVectorizer(sparse=False)
    with pytest.raises(TypeError) as error:
        v.fit(D2)
    assert str(error.value) == error_value