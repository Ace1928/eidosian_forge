import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
@pytest.mark.parametrize('strategy, expected_inv', [('uniform', [[-1.5, 2.0, -3.5, -0.5], [-0.5, 3.0, -2.5, -0.5], [0.5, 4.0, -1.5, 0.5], [0.5, 4.0, -1.5, 1.5]]), ('kmeans', [[-1.375, 2.125, -3.375, -0.5625], [-1.375, 2.125, -3.375, -0.5625], [-0.125, 3.375, -2.125, 0.5625], [0.75, 4.25, -1.25, 1.625]]), ('quantile', [[-1.5, 2.0, -3.5, -0.75], [-0.5, 3.0, -2.5, 0.0], [0.5, 4.0, -1.5, 1.25], [0.5, 4.0, -1.5, 1.25]])])
@pytest.mark.filterwarnings('ignore:In version 1.5 onwards, subsample=200_000')
@pytest.mark.parametrize('encode', ['ordinal', 'onehot', 'onehot-dense'])
def test_inverse_transform(strategy, encode, expected_inv):
    kbd = KBinsDiscretizer(n_bins=3, strategy=strategy, encode=encode)
    Xt = kbd.fit_transform(X)
    Xinv = kbd.inverse_transform(Xt)
    assert_array_almost_equal(expected_inv, Xinv)