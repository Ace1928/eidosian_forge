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
def power_special_cases(op):
    """
    Checks special cases of the uncertainty power operator op (where
    op is typically the built-in pow or uncertainties.umath.pow).

    The values x = 0, x = 1 and x = NaN are special, as are null,
    integral and NaN values of p.
    """
    zero = ufloat(0, 0)
    one = ufloat(1, 0)
    p = ufloat(0.3, 0.01)
    assert op(0, p) == 0
    assert op(zero, p) == 0
    assert op(float('nan'), zero) == 1.0
    assert op(one, float('nan')) == 1.0
    assert op(p, 0) == 1.0
    assert op(zero, 0) == 1.0
    assert op(-p, 0) == 1.0
    assert op(-10.3, zero) == 1.0
    assert op(0, zero) == 1.0
    assert op(0.3, zero) == 1.0
    assert op(-p, zero) == 1.0
    assert op(zero, zero) == 1.0
    assert op(p, zero) == 1.0
    assert op(one, -3) == 1.0
    assert op(one, -3.1) == 1.0
    assert op(one, 0) == 1.0
    assert op(one, 3) == 1.0
    assert op(one, 3.1) == 1.0
    assert op(one, -p) == 1.0
    assert op(one, zero) == 1.0
    assert op(one, p) == 1.0
    assert op(1.0, -p) == 1.0
    assert op(1.0, zero) == 1.0
    assert op(1.0, p) == 1.0