import warnings
import re
import sys
import pickle
from pathlib import Path
import os
import json
import platform
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import numpy
import numpy as np
from numpy import typecodes, array
from numpy.lib.recfunctions import rec_append_fields
from scipy import special
from scipy._lib._util import check_random_state
from scipy.integrate import (IntegrationWarning, quad, trapezoid,
import scipy.stats as stats
from scipy.stats._distn_infrastructure import argsreduce
import scipy.stats.distributions
from scipy.special import xlogy, polygamma, entr
from scipy.stats._distr_params import distcont, invdistcont
from .test_discrete_basic import distdiscrete, invdistdiscrete
from scipy.stats._continuous_distns import FitDataError, _argus_phi
from scipy.optimize import root, fmin, differential_evolution
from itertools import product
@pytest.mark.slow
@pytest.mark.xfail_on_32bit('intermittent RuntimeWarning: invalid value.')
def test_moment_vectorization(self):
    with np.errstate(invalid='ignore'):
        m = stats.studentized_range._munp([1, 2], [4, 5], [10, 11])
    assert_allclose(m.shape, (2,))
    with pytest.raises(ValueError, match='...could not be broadcast...'):
        stats.studentized_range._munp(1, [4, 5], [10, 11, 12])