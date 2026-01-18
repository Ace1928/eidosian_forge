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
@pytest.mark.parametrize('param', [0, 1])
@pytest.mark.parametrize('case', ['pdf', 'cdf'])
def test_location_scale(self, nolan_loc_scale_sample_data, param, case):
    """Tests for pdf and cdf where loc, scale are different from 0, 1
        """
    uname = platform.uname()
    is_linux_32 = uname.system == 'Linux' and '32bit' in platform.architecture()[0]
    if is_linux_32 and case == 'pdf':
        pytest.skip('Test unstable on some platforms; see gh-17839, 17859')
    data = nolan_loc_scale_sample_data
    stats.levy_stable.cdf_default_method = 'piecewise'
    stats.levy_stable.pdf_default_method = 'piecewise'
    subdata = data[data['param'] == param]
    stats.levy_stable.parameterization = f'S{param}'
    assert case in ['pdf', 'cdf']
    function = stats.levy_stable.pdf if case == 'pdf' else stats.levy_stable.cdf
    v1 = function(subdata['x'], subdata['alpha'], subdata['beta'], scale=2, loc=3)
    assert_allclose(v1, subdata[case], 1e-05)