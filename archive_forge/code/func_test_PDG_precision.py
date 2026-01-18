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
def test_PDG_precision():
    """
    Test of the calculation of the number of significant digits for
    the uncertainty.
    """
    tests = {1.7976931348623157e+308: (2, 1.7976931348623157e+308), 5e+307: (1, 5e+307), 9.976931348623156e+307: (2, 1e+308), 1.5e-323: (2, 1.5e-323), 5e-324: (1, 5e-324), 1e-323: (2, 1e-323)}
    for std_dev, result in tests.items():
        assert uncert_core.PDG_precision(std_dev) == result