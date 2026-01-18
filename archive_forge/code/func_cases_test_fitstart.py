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
def cases_test_fitstart():
    for distname, shapes in dict(distcont).items():
        if not isinstance(distname, str) or distname in {'studentized_range', 'recipinvgauss'}:
            continue
        yield (distname, shapes)