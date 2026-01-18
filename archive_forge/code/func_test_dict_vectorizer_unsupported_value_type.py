from random import Random
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
def test_dict_vectorizer_unsupported_value_type():
    """Check that we raise an error when the value associated to a feature
    is not supported.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19489
    """

    class A:
        pass
    vectorizer = DictVectorizer(sparse=True)
    X = [{'foo': A()}]
    err_msg = 'Unsupported value Type'
    with pytest.raises(TypeError, match=err_msg):
        vectorizer.fit_transform(X)