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
@pytest.mark.parametrize('u_resolution', bad_u_resolution)
def test_bad_u_resolution(self, u_resolution):
    msg = '`u_resolution` must be between 1e-15 and 1e-5.'
    with pytest.raises(ValueError, match=msg):
        NumericalInversePolynomial(StandardNormal(), u_resolution=u_resolution)