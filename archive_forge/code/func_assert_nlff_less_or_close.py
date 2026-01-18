import os
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import stats
from scipy.optimize import differential_evolution
from .test_continuous_basic import distcont
from scipy.stats._distn_infrastructure import FitError
from scipy.stats._distr_params import distdiscrete
from scipy.stats import goodness_of_fit
def assert_nlff_less_or_close(dist, data, params1, params0, rtol=1e-07, atol=0, nlff_name='nnlf'):
    nlff = getattr(dist, nlff_name)
    nlff1 = nlff(params1, data)
    nlff0 = nlff(params0, data)
    if not nlff1 < nlff0:
        np.testing.assert_allclose(nlff1, nlff0, rtol=rtol, atol=atol)