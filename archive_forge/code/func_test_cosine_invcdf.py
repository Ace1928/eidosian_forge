import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special._ufuncs import _cosine_cdf, _cosine_invcdf
@pytest.mark.parametrize('p, expected', _cosinvcdf_close)
def test_cosine_invcdf(p, expected):
    assert_allclose(_cosine_invcdf(p), expected, rtol=1e-14)