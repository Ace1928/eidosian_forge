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
def test_distributions_submodule():
    actual = set(scipy.stats.distributions.__all__)
    continuous = [dist[0] for dist in distcont]
    discrete = [dist[0] for dist in distdiscrete]
    other = ['rv_discrete', 'rv_continuous', 'rv_histogram', 'entropy', 'trapz']
    expected = continuous + discrete + other
    expected = set(filter(lambda s: not str(s).startswith('<'), expected))
    assert actual == expected