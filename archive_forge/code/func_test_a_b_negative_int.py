import sys
import pytest
import numpy as np
from typing import NamedTuple
from numpy.testing import assert_allclose
from scipy.special import hyp2f1
from scipy.special._testutils import check_version, MissingModule
@pytest.mark.parametrize('hyp2f1_test_case', [pytest.param(Hyp2f1TestCase(a=-4, b=2.02764642551431, c=1.0561196186065624, z=0.9473684210526314 - 0.10526315789473695j, expected=0.0031961077109535375 - 0.0011313924606557173j, rtol=1e-12)), pytest.param(Hyp2f1TestCase(a=-8, b=-7.937789122896016, c=-15.964218273004214, z=2 - 0.10526315789473695j, expected=0.005543763196412503 - 0.0025948879065698306j, rtol=5e-13)), pytest.param(Hyp2f1TestCase(a=-8, b=8.095813935368371, c=4.0013768449590685, z=0.9473684210526314 - 0.10526315789473695j, expected=-0.0003054674127221263 - 9.261359291755414e-05j, rtol=1e-10)), pytest.param(Hyp2f1TestCase(a=-4, b=-3.956227226099288, c=-3.9316537064827854, z=1.1578947368421053 - 0.3157894736842106j, expected=-0.0020809502580892937 - 0.0041877333232365095j, rtol=5e-12)), pytest.param(Hyp2f1TestCase(a=2.02764642551431, b=-4, c=2.050308316530781, z=0.9473684210526314 - 0.10526315789473695j, expected=0.0011282435590058734 + 0.0002027062303465851j, rtol=5e-13)), pytest.param(Hyp2f1TestCase(a=-7.937789122896016, b=-8, c=-15.964218273004214, z=1.3684210526315788 + 0.10526315789473673j, expected=-9.134907719238265e-05 - 0.00040219233987390723j, rtol=5e-12)), pytest.param(Hyp2f1TestCase(a=4.080187217753502, b=-4, c=4.0013768449590685, z=0.9473684210526314 - 0.10526315789473695j, expected=-0.000519013062087489 - 0.0005855883076830948j, rtol=5e-12)), pytest.param(Hyp2f1TestCase(a=-10000, b=2.2, c=93459345.3, z=2 + 2j, expected=0.9995292071559088 - 0.00047047067522659253j, rtol=1e-12))])
def test_a_b_negative_int(self, hyp2f1_test_case):
    a, b, c, z, expected, rtol = hyp2f1_test_case
    assert a == int(a) and a < 0 or (b == int(b) and b < 0)
    assert_allclose(hyp2f1(a, b, c, z), expected, rtol=rtol)