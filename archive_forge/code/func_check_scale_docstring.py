import numpy.testing as npt
from numpy.testing import assert_allclose
import numpy as np
import pytest
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distdiscrete, invdistdiscrete
from scipy.stats._distn_infrastructure import rv_discrete_frozen
def check_scale_docstring(distfn):
    if distfn.__doc__ is not None:
        npt.assert_('scale' not in distfn.__doc__)