import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
def test_errobj_noerrmask(self):
    olderrobj = np.geterrobj()
    try:
        np.seterrobj([umath.UFUNC_BUFSIZE_DEFAULT, umath.ERR_DEFAULT + 1, None])
        np.isnan(np.array([6]))
        for i in range(10000):
            np.seterrobj([umath.UFUNC_BUFSIZE_DEFAULT, umath.ERR_DEFAULT, None])
        np.isnan(np.array([6]))
    finally:
        np.seterrobj(olderrobj)