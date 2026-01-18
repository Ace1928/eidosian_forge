import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
def test_encode_options():
    est = KBinsDiscretizer(n_bins=[2, 3, 3, 3], encode='ordinal').fit(X)
    Xt_1 = est.transform(X)
    est = KBinsDiscretizer(n_bins=[2, 3, 3, 3], encode='onehot-dense').fit(X)
    Xt_2 = est.transform(X)
    assert not sp.issparse(Xt_2)
    assert_array_equal(OneHotEncoder(categories=[np.arange(i) for i in [2, 3, 3, 3]], sparse_output=False).fit_transform(Xt_1), Xt_2)
    est = KBinsDiscretizer(n_bins=[2, 3, 3, 3], encode='onehot').fit(X)
    Xt_3 = est.transform(X)
    assert sp.issparse(Xt_3)
    assert_array_equal(OneHotEncoder(categories=[np.arange(i) for i in [2, 3, 3, 3]], sparse_output=True).fit_transform(Xt_1).toarray(), Xt_3.toarray())