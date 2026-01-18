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
def test_broadcasting_in_moments_gh12192_regression():
    vals0 = stats.norm.moment(order=1, loc=np.array([1, 2, 3]), scale=[[1]])
    expected0 = np.array([[1.0, 2.0, 3.0]])
    npt.assert_equal(vals0, expected0)
    assert vals0.shape == expected0.shape
    vals1 = stats.norm.moment(order=1, loc=np.array([[1], [2], [3]]), scale=[1, 2, 3])
    expected1 = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    npt.assert_equal(vals1, expected1)
    assert vals1.shape == expected1.shape
    vals2 = stats.chi.moment(order=1, df=[1.0, 2.0, 3.0], loc=0.0, scale=1.0)
    expected2 = np.array([0.79788456, 1.25331414, 1.59576912])
    npt.assert_allclose(vals2, expected2, rtol=1e-08)
    assert vals2.shape == expected2.shape
    vals3 = stats.chi.moment(order=1, df=[[1.0], [2.0], [3.0]], loc=[0.0, 1.0, 2.0], scale=[-1.0, 0.0, 3.0])
    expected3 = np.array([[np.nan, np.nan, 4.39365368], [np.nan, np.nan, 5.75994241], [np.nan, np.nan, 6.78730736]])
    npt.assert_allclose(vals3, expected3, rtol=1e-08)
    assert vals3.shape == expected3.shape