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
def test_kappa3_array_gh13582():
    shapes = [0.5, 1.5, 2.5, 3.5, 4.5]
    moments = 'mvsk'
    res = np.array([[stats.kappa3.stats(shape, moments=moment) for shape in shapes] for moment in moments])
    res2 = np.array(stats.kappa3.stats(shapes, moments=moments))
    npt.assert_allclose(res, res2)