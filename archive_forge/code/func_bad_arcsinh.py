import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
def bad_arcsinh():
    """The blocklisted trig functions are not accurate on aarch64/PPC for
    complex256. Rather than dig through the actual problem skip the
    test. This should be fixed when we can move past glibc2.17
    which is the version in manylinux2014
    """
    if platform.machine() == 'aarch64':
        x = 1.78e-10
    elif on_powerpc():
        x = 2.16e-10
    else:
        return False
    v1 = np.arcsinh(np.float128(x))
    v2 = np.arcsinh(np.complex256(x)).real
    return abs(v1 / v2 - 1.0) > 1e-23