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
@pytest.mark.parametrize('construction_points', [-1, 0, 0.1])
def test_bad_construction_points_scalar(self, construction_points):
    with pytest.raises(ValueError, match='`construction_points` must be a positive integer.'):
        TransformedDensityRejection(StandardNormal(), construction_points=construction_points)