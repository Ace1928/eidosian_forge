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
@pytest.mark.parametrize('domain', inf_domain)
def test_inf_domain(self, domain):
    with pytest.raises(ValueError, match='must be finite'):
        DiscreteGuideTable(stats.binom(10, 0.2), domain=domain)