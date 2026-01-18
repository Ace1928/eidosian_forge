import sys
import pytest
import numpy as np
from typing import NamedTuple
from numpy.testing import assert_allclose
from scipy.special import hyp2f1
from scipy.special._testutils import check_version, MissingModule
@pytest.mark.parametrize('hyp2f1_test_case', [pytest.param(Hyp2f1TestCase(a=0.5, b=0.2, c=-10, z=0.2 + 0.2j, expected=np.inf + 0j, rtol=0)), pytest.param(Hyp2f1TestCase(a=0.5, b=0.2, c=-10, z=0 + 0j, expected=1 + 0j, rtol=0)), pytest.param(Hyp2f1TestCase(a=0.5, b=0, c=-10, z=0.2 + 0.2j, expected=1 + 0j, rtol=0)), pytest.param(Hyp2f1TestCase(a=0.5, b=0, c=0, z=0.2 + 0.2j, expected=1 + 0j, rtol=0)), pytest.param(Hyp2f1TestCase(a=0.5, b=0.2, c=0, z=0.2 + 0.2j, expected=np.inf + 0j, rtol=0)), pytest.param(Hyp2f1TestCase(a=0.5, b=0.2, c=0, z=0 + 0j, expected=np.nan + 0j, rtol=0)), pytest.param(Hyp2f1TestCase(a=0.5, b=-5, c=-10, z=0.2 + 0.2j, expected=1.0495404166666666 + 0.05708208333333334j, rtol=1e-15)), pytest.param(Hyp2f1TestCase(a=0.5, b=-10, c=-10, z=0.2 + 0.2j, expected=1.092966013125 + 0.13455014673750001j, rtol=1e-15)), pytest.param(Hyp2f1TestCase(a=-10, b=-20, c=-10, z=0.2 + 0.2j, expected=-0.07712512000000005 + 0.12752814080000005j, rtol=1e-13)), pytest.param(Hyp2f1TestCase(a=-1, b=3.2, c=-1, z=0.2 + 0.2j, expected=1.6400000000000001 + 0.6400000000000001j, rtol=1e-13)), pytest.param(Hyp2f1TestCase(a=-2, b=1.2, c=-4, z=1 + 0j, expected=1.82 + 0j, rtol=1e-15))])
def test_c_non_positive_int(self, hyp2f1_test_case):
    a, b, c, z, expected, rtol = hyp2f1_test_case
    assert_allclose(hyp2f1(a, b, c, z), expected, rtol=rtol)