import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_allclose(self):
    """Test for false positive on allclose in normalize() in
        filter_design.py"""
    b_matlab = np.array([2.150733144728282e-11, 1.720586515782626e-10, 6.02205280523919e-10, 1.204410561047838e-09, 1.505513201309798e-09, 1.204410561047838e-09, 6.02205280523919e-10, 1.720586515782626e-10, 2.150733144728282e-11])
    a_matlab = np.array([1.0, -7.782402035027959, 26.54354569747454, -51.82182531666387, 63.34127355102684, -49.63358186631157, 24.34862182949389, -6.836925348604676, 0.841293494444914])
    b_norm_in = np.array([1.5543135865293012e-06, 1.2434508692234413e-05, 4.352078042282045e-05, 8.70415608456409e-05, 0.00010880195105705122, 8.704156084564097e-05, 4.352078042282045e-05, 1.2434508692234413e-05, 1.5543135865293012e-06])
    a_norm_in = np.array([72269.02590912717, -562426.6143046797, 1918276.1917308895, -3745112.8364682454, 4577612.139376277, -3586970.6138592605, 1759651.1818472347, -494097.93515707983, 60799.46134721965])
    b_output, a_output = normalize(b_norm_in, a_norm_in)
    assert_array_almost_equal(b_matlab, b_output, decimal=13)
    assert_array_almost_equal(a_matlab, a_output, decimal=13)