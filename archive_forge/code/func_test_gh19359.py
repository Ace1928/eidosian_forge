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
def test_gh19359(self):
    pv = special.softmax(np.ones((1533,)))
    rng = DiscreteAliasUrn(pv, random_state=42)
    check_discr_samples(rng, pv, (1532 / 2, (1532 ** 2 - 1) / 12), rtol=0.005)