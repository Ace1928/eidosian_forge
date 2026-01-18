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
@pytest.fixture
def nolan_cdf_sample_data(self):
    """Sample data points for cdf computed with Nolan's stablec

        See - http://fs2.american.edu/jpnolan/www/stable/stable.html

        There's a known limitation of Nolan's executable for alpha < 0.2.

        The data table loaded below is generated from Nolan's stablec
        with the following parameter space:

            alpha = 0.1, 0.2, ..., 2.0
            beta = -1.0, -0.9, ..., 1.0
            p = 0.01, 0.05, 0.1, 0.25, 0.35, 0.5,

        and the equivalent for the right tail

        Ideally, Nolan's output for CDF values should match the percentile
        from where they have been sampled from. Even more so as we extract
        percentile x positions from stablec too. However, we note at places
        Nolan's stablec will produce absolute errors in order of 1e-5. We
        compare against his calculations here. In future, once we less
        reliant on Nolan's paper we might switch to comparing directly at
        percentiles (those x values being produced from some alternative
        means).

        Typically inputs for stablec:

            stablec.exe <<
            2 # cdf
            1 # Nolan S equivalent to S0 in scipy
            .25,2,.25 # alpha
            -1,-1,0 # beta
            -10,10,1 # x
            1,0 # gamma, delta
            2 # output file
        """
    data = np.load(Path(__file__).parent / 'data/levy_stable/stable-Z1-cdf-sample-data.npy')
    data = np.rec.fromarrays(data.T, names='x,p,alpha,beta,pct')
    return data