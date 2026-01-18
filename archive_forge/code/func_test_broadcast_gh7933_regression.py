import sys
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises as assert_raises
from scipy.integrate import IntegrationWarning
import itertools
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen
def test_broadcast_gh7933_regression():
    stats.truncnorm.logpdf(np.array([3.0, 2.0, 1.0]), a=(1.5 - np.array([6.0, 5.0, 4.0])) / 3.0, b=np.inf, loc=np.array([6.0, 5.0, 4.0]), scale=3.0)