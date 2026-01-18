import sys
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises as assert_raises
from scipy.integrate import IntegrationWarning
import itertools
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen
def check_loc_scale(distfn, arg, m, v, msg):
    loc, scale = (np.array([10.0, 20.0]), np.array([10.0, 20.0]))
    mt, vt = distfn.stats(*arg, loc=loc, scale=scale)
    npt.assert_allclose(m * scale + loc, mt)
    npt.assert_allclose(v * scale * scale, vt)