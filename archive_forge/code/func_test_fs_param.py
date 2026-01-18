import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_fs_param(self):
    b, a = butter(4, 4800, fs=96000)
    w = np.linspace(0, 96000 / 2, num=10, endpoint=False)
    w, gd = group_delay((b, a), w=w, fs=96000)
    norm_gd = np.array([8.249313898506037, 11.958947880907104, 2.452325615326005, 1.048918665702008, 0.611382575635897, 0.418293269460578, 0.317932917836572, 0.261371844762525, 0.229038045801298, 0.212185774208521])
    assert_array_almost_equal(gd, norm_gd)