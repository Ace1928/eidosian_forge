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
def test_guide_factor_zero_raises_warning(self):
    pv = [0.1, 0.3, 0.6]
    urng = np.random.default_rng()
    with pytest.warns(RuntimeWarning):
        DiscreteGuideTable(pv, random_state=urng, guide_factor=0)