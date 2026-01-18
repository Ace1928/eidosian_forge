from __future__ import division
from __future__ import absolute_import
import sys
import math
from uncertainties import ufloat
import uncertainties.core as uncert_core
import uncertainties.umath_core as umath_core
from . import test_uncertainties
def test_compound_expression():
    """
    Test equality between different formulas.
    """
    x = ufloat(3, 0.1)
    assert umath_core.tan(x) == umath_core.sin(x) / umath_core.cos(x)