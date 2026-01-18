import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def check_categorical_onehot(X):
    enc = OneHotEncoder(categories='auto')
    Xtr1 = enc.fit_transform(X)
    enc = OneHotEncoder(categories='auto', sparse_output=False)
    Xtr2 = enc.fit_transform(X)
    assert_allclose(Xtr1.toarray(), Xtr2)
    assert sparse.issparse(Xtr1) and Xtr1.format == 'csr'
    return Xtr1.toarray()