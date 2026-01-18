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
def check_sample_var(sample, popvar):
    res = stats.bootstrap((sample,), lambda x, axis: x.var(ddof=1, axis=axis), confidence_level=0.995)
    conf = res.confidence_interval
    low, high = (conf.low, conf.high)
    assert low <= popvar <= high