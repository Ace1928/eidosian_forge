import functools
import itertools
import math
import numpy
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from scipy.ndimage._filters import _gaussian_kernel1d
from . import types, float_types, complex_types
def test_gaussian_kernel1d():
    radius = 10
    sigma = 2
    sigma2 = sigma * sigma
    x = numpy.arange(-radius, radius + 1, dtype=numpy.double)
    phi_x = numpy.exp(-0.5 * x * x / sigma2)
    phi_x /= phi_x.sum()
    assert_allclose(phi_x, _gaussian_kernel1d(sigma, 0, radius))
    assert_allclose(-phi_x * x / sigma2, _gaussian_kernel1d(sigma, 1, radius))
    assert_allclose(phi_x * (x * x / sigma2 - 1) / sigma2, _gaussian_kernel1d(sigma, 2, radius))
    assert_allclose(phi_x * (3 - x * x / sigma2) * x / (sigma2 * sigma2), _gaussian_kernel1d(sigma, 3, radius))