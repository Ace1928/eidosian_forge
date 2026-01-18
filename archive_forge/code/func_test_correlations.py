from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import zip
from builtins import map
from builtins import range
import copy
import weakref
import math
from math import isnan, isinf
import random
import sys
import uncertainties.core as uncert_core
from uncertainties.core import ufloat, AffineScalarFunc, ufloat_fromstr
from uncertainties import umath
def test_correlations():
    """Correlations between variables"""
    a = ufloat(1, 0)
    x = ufloat(4, 0.1)
    y = x * 2 + a
    assert y.std_dev != 0
    normally_zero = y - (x * 2 + 1)
    assert normally_zero.nominal_value == 0
    assert normally_zero.std_dev == 0