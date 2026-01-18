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
def test_negative_cdf_bug_11186(self):
    skews = [-3, -1, 0, 0.5]
    x_eval = 0.5
    neg_inf = -30
    cdfs = stats.pearson3.cdf(x_eval, skews)
    int_pdfs = [quad(stats.pearson3(skew).pdf, neg_inf, x_eval)[0] for skew in skews]
    assert_allclose(cdfs, int_pdfs)