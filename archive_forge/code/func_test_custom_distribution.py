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
def test_custom_distribution(self):
    dist1 = StandardNormal()
    fni1 = NumericalInverseHermite(dist1)
    dist2 = stats.norm()
    fni2 = NumericalInverseHermite(dist2)
    assert_allclose(fni1.rvs(random_state=0), fni2.rvs(random_state=0))