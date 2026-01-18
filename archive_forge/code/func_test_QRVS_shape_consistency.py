import threading
import pickle
import pytest
from copy import deepcopy
import platform
import sys
import math
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy.stats.sampling import (
from pytest import raises as assert_raises
from scipy import stats
from scipy import special
from scipy.stats import chisquare, cramervonmises
from scipy.stats._distr_params import distdiscrete, distcont
from scipy._lib._util import check_random_state
@pytest.mark.parametrize('qrng', qrngs)
@pytest.mark.parametrize('size_in, size_out', sizes)
@pytest.mark.parametrize('d_in, d_out', ds)
def test_QRVS_shape_consistency(self, qrng, size_in, size_out, d_in, d_out, method):
    w32 = sys.platform == 'win32' and platform.architecture()[0] == '32bit'
    if w32 and method == 'NumericalInversePolynomial':
        pytest.xfail('NumericalInversePolynomial.qrvs fails for Win 32-bit')
    dist = StandardNormal()
    Method = getattr(stats.sampling, method)
    gen = Method(dist)
    if d_in is not None and qrng is not None and (qrng.d != d_in):
        match = '`d` must be consistent with dimension of `qmc_engine`.'
        with pytest.raises(ValueError, match=match):
            gen.qrvs(size_in, d=d_in, qmc_engine=qrng)
        return
    if d_in is None and qrng is not None and (qrng.d != 1):
        d_out = (qrng.d,)
    shape_expected = size_out + d_out
    qrng2 = deepcopy(qrng)
    qrvs = gen.qrvs(size=size_in, d=d_in, qmc_engine=qrng)
    if size_in is not None:
        assert qrvs.shape == shape_expected
    if qrng2 is not None:
        uniform = qrng2.random(np.prod(size_in) or 1)
        qrvs2 = stats.norm.ppf(uniform).reshape(shape_expected)
        assert_allclose(qrvs, qrvs2, atol=1e-12)