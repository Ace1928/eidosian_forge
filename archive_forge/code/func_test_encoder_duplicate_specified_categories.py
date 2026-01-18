import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('Encoder', [OneHotEncoder, OrdinalEncoder])
def test_encoder_duplicate_specified_categories(Encoder):
    """Test encoder for specified categories have duplicate values.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27088
    """
    cats = [np.array(['a', 'b', 'a'], dtype=object)]
    enc = Encoder(categories=cats)
    X = np.array([['a', 'b']], dtype=object).T
    with pytest.raises(ValueError, match='the predefined categories contain duplicate elements.'):
        enc.fit(X)