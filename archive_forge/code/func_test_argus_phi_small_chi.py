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
@pytest.mark.parametrize('chi, expected, rtol', [(0.9, 0.07646314974436118, 1e-14), (0.5, 0.015429797891863365, 1e-14), (0.1, 0.0001325825293278049, 1e-14), (0.01, 1.3297677078224565e-07, 1e-15), (0.001, 1.3298072023958999e-10, 1e-14), (0.0001, 1.3298075973486862e-13, 1e-14), (1e-06, 1.32980760133771e-19, 1e-14), (1e-09, 1.329807601338109e-28, 1e-15)])
def test_argus_phi_small_chi(self, chi, expected, rtol):
    assert_allclose(_argus_phi(chi), expected, rtol=rtol)