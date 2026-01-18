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
def check_ppf_private(distfn, arg, msg):
    ppfs = distfn._ppf(np.array([0.1, 0.5, 0.9]), *arg)
    npt.assert_(not np.any(np.isnan(ppfs)), msg + 'ppf private is nan')