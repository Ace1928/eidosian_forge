import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special._ufuncs import _cosine_cdf, _cosine_invcdf
@pytest.mark.parametrize('x, expected', _coscdf_close)
def test_cosine_cdf(x, expected):
    assert_allclose(_cosine_cdf(x), expected, rtol=5e-15)