from numpy.testing import (assert_, assert_equal, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
from numpy import mgrid, pi, sin, ogrid, poly1d, linspace
import numpy as np
from scipy.interpolate import (interp1d, interp2d, lagrange, PPoly, BPoly,
from scipy.special import poch, gamma
from scipy.interpolate import _ppoly
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
from scipy.integrate import nquad
from scipy.special import binom
def binom_matrix(power):
    n = np.arange(power + 1).reshape(-1, 1)
    k = np.arange(power + 1)
    B = binom(n, k)
    return B[::-1, ::-1]