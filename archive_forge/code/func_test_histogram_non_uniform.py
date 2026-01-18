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
def test_histogram_non_uniform():
    counts, bins = ([1, 1], [0, 1, 1001])
    dist = stats.rv_histogram((counts, bins), density=False)
    np.testing.assert_allclose(dist.pdf([0.5, 200]), [0.5, 0.0005])
    assert dist.median() == 1
    dist = stats.rv_histogram((counts, bins), density=True)
    np.testing.assert_allclose(dist.pdf([0.5, 200]), 1 / 1001)
    assert dist.median() == 1001 / 2
    message = 'Bin widths are not constant. Assuming...'
    with assert_warns(RuntimeWarning, match=message):
        dist = stats.rv_histogram((counts, bins))
        assert dist.median() == 1001 / 2
    dist = stats.rv_histogram((counts, [0, 1, 2]))
    assert dist.median() == 1