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
def check_cdf_sf(distfn, arg, msg):
    npt.assert_almost_equal(distfn.cdf([0.1, 0.9], *arg), 1.0 - distfn.sf([0.1, 0.9], *arg), decimal=DECIMAL, err_msg=msg + ' - cdf-sf relationship')