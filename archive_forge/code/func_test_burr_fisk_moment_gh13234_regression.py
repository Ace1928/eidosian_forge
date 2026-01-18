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
def test_burr_fisk_moment_gh13234_regression():
    vals0 = stats.burr.moment(1, 5, 4)
    assert isinstance(vals0, float)
    vals1 = stats.fisk.moment(1, 8)
    assert isinstance(vals1, float)