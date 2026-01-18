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
def test_access_to_std_dev():
    """Uniform access to the standard deviation"""
    x = ufloat(1, 0.1)
    y = 2 * x
    assert uncert_core.std_dev(x) == x.std_dev
    assert uncert_core.std_dev(y) == y.std_dev
    assert uncert_core.std_dev([]) == 0
    assert uncert_core.std_dev(None) == 0