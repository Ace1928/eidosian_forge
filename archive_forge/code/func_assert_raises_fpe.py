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
def assert_raises_fpe(self, fpeerr, flop, x, y):
    ftype = type(x)
    try:
        flop(x, y)
        assert_(False, "Type %s did not raise fpe error '%s'." % (ftype, fpeerr))
    except FloatingPointError as exc:
        assert_(str(exc).find(fpeerr) >= 0, "Type %s raised wrong fpe error '%s'." % (ftype, exc))