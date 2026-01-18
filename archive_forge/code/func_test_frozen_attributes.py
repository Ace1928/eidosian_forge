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
def test_frozen_attributes():
    message = "'rv_continuous_frozen' object has no attribute"
    with pytest.raises(AttributeError, match=message):
        stats.norm().pmf
    with pytest.raises(AttributeError, match=message):
        stats.norm().logpmf
    stats.norm.pmf = 'herring'
    frozen_norm = stats.norm()
    assert isinstance(frozen_norm, rv_continuous_frozen)
    delattr(stats.norm, 'pmf')