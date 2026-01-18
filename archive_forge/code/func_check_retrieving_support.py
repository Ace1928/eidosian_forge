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
def check_retrieving_support(distfn, args):
    loc, scale = (1, 2)
    supp = distfn.support(*args)
    supp_loc_scale = distfn.support(*args, loc=loc, scale=scale)
    npt.assert_almost_equal(np.array(supp) * scale + loc, np.array(supp_loc_scale))