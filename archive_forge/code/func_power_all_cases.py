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
def power_all_cases(op):
    """
    Checks all cases for the value and derivatives of power-like
    operator op (op is typically the built-in pow(), or math.pow()).

    Checks only the details of special results like 0, 1 or NaN).

    Different cases for the value of x**p and its derivatives are
    tested by dividing the (x, p) plane with:

    - x < 0, x = 0, x > 0
    - p integer or not, p < 0, p = 0, p > 0

    (not all combinations are distinct: for instance x > 0 gives
    identical formulas for all p).
    """
    zero = ufloat(0, 0.1)
    zero2 = ufloat(0, 0.1)
    one = ufloat(1, 0.1)
    positive = ufloat(0.3, 0.01)
    positive2 = ufloat(0.3, 0.01)
    negative = ufloat(-0.3, 0.01)
    integer = ufloat(-3, 0)
    non_int_larger_than_one = ufloat(3.1, 0.01)
    positive_smaller_than_one = ufloat(0.3, 0.01)
    result = op(negative, integer)
    assert not isnan(result.derivatives[negative])
    assert isnan(result.derivatives[integer])
    result = op(negative, one)
    assert result.derivatives[negative] == 1
    assert isnan(result.derivatives[one])
    result = op(negative, zero)
    assert result.derivatives[negative] == 0
    assert isnan(result.derivatives[zero])
    result = op(zero, non_int_larger_than_one)
    assert isnan(result.derivatives[zero])
    assert result.derivatives[non_int_larger_than_one] == 0
    result = op(zero, one)
    assert result.derivatives[zero] == 1
    assert result.derivatives[one] == 0
    result = op(zero, 2 * one)
    assert result.derivatives[zero] == 0
    assert result.derivatives[one] == 0
    result = op(zero, positive_smaller_than_one)
    assert isnan(result.derivatives[zero])
    assert result.derivatives[positive_smaller_than_one] == 0
    result = op(zero, zero2)
    assert result.derivatives[zero] == 0
    assert isnan(result.derivatives[zero2])
    result = op(positive, positive2)
    assert not isnan(result.derivatives[positive])
    assert not isnan(result.derivatives[positive2])
    result = op(positive, zero)
    assert result.derivatives[positive] == 0
    assert not isnan(result.derivatives[zero])
    result = op(positive, negative)
    assert not isnan(result.derivatives[positive])
    assert not isnan(result.derivatives[negative])